#training StyleGANv2, Grad-Cam
import os
import torch
import torch.nn as nn
import torchvision
import model.E_v3_BIG as BE
from model.utils.custom_adam import LREQAdam
from model.utils.biggan_config import BigGANConfig
from model.biggan.biggan_generator import BigGAN
import lpips
import metric.pytorch_ssim as pytorch_ssim
from metric.grad_cam import GradCAM, GradCamPlusPlus, GuidedBackPropagation, mask2cam
import tensorboardX
import numpy as np
#torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False # faster

# 1.难度优化: 输出的label来一组 argmax()再onehot
# 2.网络中的IN改为CBN
# 3.condVector加入latent_space
# 4.降低训练难度，每一轮标签统一
# 5.进一步降低难度，不在预测标签
# -.网络的Z改为W


def one_hot(x, class_count=1000):
    # 第一构造一个[class_count, class_count]的对角线为1的向量
    # 第二保留label对应的行并返回
    return torch.eye(class_count)[x,:]

from scipy.stats import truncnorm
def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

def set_seed(seed): #随机数设置
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

def space_loss(imgs1,imgs2,image_space=True,lpips_model=None):
    loss_mse = torch.nn.MSELoss()
    loss_kl = torch.nn.KLDivLoss()
    ssim_loss = pytorch_ssim.SSIM()
    loss_lpips = lpips_model

    imgs1 = imgs1.contiguous()
    imgs2 = imgs2.contiguous()

    loss_imgs_mse_1 = loss_mse(imgs1,imgs2)
    loss_imgs_mse_2 = loss_mse(imgs1.mean(),imgs2.mean())
    loss_imgs_mse_3 = loss_mse(imgs1.std(),imgs2.std())
    loss_imgs_mse = loss_imgs_mse_1 + loss_imgs_mse_2 + loss_imgs_mse_3

    imgs1_kl, imgs2_kl = torch.nn.functional.softmax(imgs1),torch.nn.functional.softmax(imgs2)
    loss_imgs_kl = loss_kl(torch.log(imgs2_kl),imgs1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
    loss_imgs_kl = torch.where(torch.isnan(loss_imgs_kl),torch.full_like(loss_imgs_kl,0), loss_imgs_kl)
    loss_imgs_kl = torch.where(torch.isinf(loss_imgs_kl),torch.full_like(loss_imgs_kl,1), loss_imgs_kl)

    imgs1_cos = imgs1.view(-1)
    imgs2_cos = imgs2.view(-1)
    loss_imgs_cosine = 1 - imgs1_cos.dot(imgs2_cos)/(torch.sqrt(imgs1_cos.dot(imgs1_cos))*torch.sqrt(imgs2_cos.dot(imgs2_cos))) #[-1,1],-1:反向相反，1:方向相同

    if  image_space:
        ssim_value = pytorch_ssim.ssim(imgs1, imgs2) # while ssim_value<0.999:
        loss_imgs_ssim = 1-ssim_loss(imgs1, imgs2)
    else:
        loss_imgs_ssim = torch.tensor(0)

    if image_space:
        loss_imgs_lpips = loss_lpips(imgs1,imgs2).mean()
    else:
        loss_imgs_lpips = torch.tensor(0)

    loss_imgs = loss_imgs_mse + loss_imgs_kl + loss_imgs_cosine + loss_imgs_ssim + loss_imgs_lpips
    loss_info = [[loss_imgs_mse_1.item(),loss_imgs_mse_2.item(),loss_imgs_mse_3.item()], loss_imgs_kl.item(), loss_imgs_cosine.item(), loss_imgs_ssim.item(), loss_imgs_lpips.item()]
    return loss_imgs, loss_info

def train(generator = None, tensor_writer = None, synthesis_kwargs = None):
    G = generator
    E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3, biggan=True)
    #E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/EAE-car-cat/result/EB_cat_cosine_v2/E_model_ep80000.pth'))
    G.cuda()
    E.cuda()
    writer = tensor_writer

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0) 
    #用这个adam不会报错:RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

    batch_size = 4

    it_d = 0

    vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
    final_layer = None
    for name, m in vgg16.named_modules():
        if isinstance(m, nn.Conv2d):
            final_layer = name
    grad_cam_plus_plus = GradCamPlusPlus(vgg16, final_layer)
    gbp = GuidedBackPropagation(vgg16)

    it_d = 0
    for epoch in range(0,250001):
        set_seed(epoch%30000)
        z = truncated_noise_sample(truncation=synthesis_kwargs, batch_size=batch_size, seed=epoch%30000)
        #label = np.random.randint(1000,size=batch_size) # 生成标签
        flag = np.random.randint(1000)
        label = np.ones(batch_size)
        label = flag * label
        label = one_hot(label)
        z = torch.tensor(z, dtype=torch.float).cuda()
        w1 = torch.tensor(label, dtype=torch.float).cuda()
        truncation = torch.tensor(synthesis_kwargs, dtype=torch.float).cuda()
        with torch.no_grad(): #这里需要生成图片和变量
            imgs1, cond_vector = G(z, w1, truncation)

        z2, cond_vector2 = E(imgs1.cuda(), cond_vector)
        #w2_ = w2.argmax(dim=1)
        #w2_ = one_hot(w2_).requires_grad_(True).cuda()
        imgs2, _=G(z2, w1, truncation)
        
        E_optimizer.zero_grad()

#Latent Space
    ##--C
        loss_c, loss_c_info = space_loss(z,z2,image_space = False)
        E_optimizer.zero_grad()
        loss_c.backward(retain_graph=True)
        E_optimizer.step()

    ##--W
        loss_w, loss_w_info = space_loss(cond_vector,cond_vector2,image_space = False)
        E_optimizer.zero_grad()
        loss_w.backward(retain_graph=True)
        E_optimizer.step()

    # ##--cond_vector
    #     loss_condVector, loss_condVector_info = space_loss(cond_vector,cond_vector2,image_space = False)
    #     E_optimizer.zero_grad()
    #     loss_condVector.backward(retain_graph=True)
    #     E_optimizer.step()

#Image Space
        mask_1 = grad_cam_plus_plus(imgs1,None) #[c,1,h,w]
        mask_2 = grad_cam_plus_plus(imgs2,None)
        #imgs1.retain_grad()
        #imgs2.retain_grad()
        imgs1_ = imgs1.detach().clone()
        imgs1_.requires_grad = True
        imgs2_ = imgs2.detach().clone()
        imgs2_.requires_grad = True
        grad_1 = gbp(imgs1_) # [n,c,h,w]
        grad_2 = gbp(imgs2_)
        heatmap_1,cam_1 = mask2cam(mask_1,imgs1)
        heatmap_2,cam_2 = mask2cam(mask_2,imgs2)

    ##--Mask_Cam
        mask_1 = mask_1.cuda().float()
        mask_1.requires_grad=True
        mask_2 = mask_2.cuda().float()
        mask_2.requires_grad=True
        loss_mask, loss_mask_info = space_loss(mask_1,mask_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_mask.backward(retain_graph=True)
        E_optimizer.step()

    ##--Grad
        grad_1 = grad_1.cuda().float()
        grad_1.requires_grad=True
        grad_2 = grad_2.cuda().float()
        grad_2.requires_grad=True
        loss_grad, loss_grad_info = space_loss(grad_1,grad_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_grad.backward(retain_graph=True)
        E_optimizer.step()

    ##--Image
        loss_imgs, loss_imgs_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        loss_imgs.backward(retain_graph=True)
        E_optimizer.step()

    ##--Grad_CAM from mask
        cam_1 = cam_1.cuda().float()
        cam_1.requires_grad=True
        cam_2 = cam_2.cuda().float()
        cam_2.requires_grad=True
        loss_Gcam, loss_Gcam_info = space_loss(cam_1,cam_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_Gcam.backward(retain_graph=True)
        E_optimizer.step()

        print('i_'+str(epoch))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_ssim, loss_imgs_cosine, loss_kl_imgs, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_mask_info: %s'%loss_mask_info)
        print('loss_grad_info: %s'%loss_grad_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('loss_Gcam_info: %s'%loss_Gcam_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c_info: %s'%loss_c_info)
        #print('loss_condVector_info: %s'%loss_condVector_info)

        it_d += 1
        writer.add_scalar('loss_mask_mse', loss_mask_info[0][0], global_step=it_d)
        writer.add_scalar('loss_mask_mse_mean', loss_mask_info[0][1], global_step=it_d)
        writer.add_scalar('loss_mask_mse_std', loss_mask_info[0][2], global_step=it_d)
        writer.add_scalar('loss_mask_kl', loss_mask_info[1], global_step=it_d)
        writer.add_scalar('loss_mask_cosine', loss_mask_info[2], global_step=it_d)
        writer.add_scalar('loss_mask_ssim', loss_mask_info[3], global_step=it_d)
        writer.add_scalar('loss_mask_lpips', loss_mask_info[4], global_step=it_d)

        writer.add_scalar('loss_grad_mse', loss_grad_info[0][0], global_step=it_d)
        writer.add_scalar('loss_grad_mse_mean', loss_grad_info[0][1], global_step=it_d)
        writer.add_scalar('loss_grad_mse_std', loss_grad_info[0][2], global_step=it_d)
        writer.add_scalar('loss_grad_kl', loss_grad_info[1], global_step=it_d)
        writer.add_scalar('loss_grad_cosine', loss_grad_info[2], global_step=it_d)
        writer.add_scalar('loss_grad_ssim', loss_grad_info[3], global_step=it_d)
        writer.add_scalar('loss_grad_lpips', loss_grad_info[4], global_step=it_d)

        writer.add_scalar('loss_imgs_mse', loss_imgs_info[0][0], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_mean', loss_imgs_info[0][1], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_std', loss_imgs_info[0][2], global_step=it_d)
        writer.add_scalar('loss_imgs_kl', loss_imgs_info[1], global_step=it_d)
        writer.add_scalar('loss_imgs_cosine', loss_imgs_info[2], global_step=it_d)
        writer.add_scalar('loss_imgs_ssim', loss_imgs_info[3], global_step=it_d)
        writer.add_scalar('loss_imgs_lpips', loss_imgs_info[4], global_step=it_d)

        writer.add_scalar('loss_Gcam', loss_Gcam_info[0][0], global_step=it_d)
        writer.add_scalar('loss_Gcam_mean', loss_Gcam_info[0][1], global_step=it_d)
        writer.add_scalar('loss_Gcam_std', loss_Gcam_info[0][2], global_step=it_d)
        writer.add_scalar('loss_Gcam_kl', loss_Gcam_info[1], global_step=it_d)
        writer.add_scalar('loss_Gcam_cosine', loss_Gcam_info[2], global_step=it_d)
        writer.add_scalar('loss_Gcam_ssim', loss_Gcam_info[3], global_step=it_d)
        writer.add_scalar('loss_Gcam_lpips', loss_Gcam_info[4], global_step=it_d)

        writer.add_scalar('loss_w_mse', loss_w_info[0][0], global_step=it_d)
        writer.add_scalar('loss_w_mse_mean', loss_w_info[0][1], global_step=it_d)
        writer.add_scalar('loss_w_mse_std', loss_w_info[0][2], global_step=it_d)
        writer.add_scalar('loss_w_kl', loss_w_info[1], global_step=it_d)
        writer.add_scalar('loss_w_cosine', loss_w_info[2], global_step=it_d)
        writer.add_scalar('loss_w_ssim', loss_w_info[3], global_step=it_d)
        writer.add_scalar('loss_w_lpips', loss_w_info[4], global_step=it_d)

        writer.add_scalar('loss_c_mse', loss_c_info[0][0], global_step=it_d)
        writer.add_scalar('loss_c_mse_mean', loss_c_info[0][1], global_step=it_d)
        writer.add_scalar('loss_c_mse_std', loss_c_info[0][2], global_step=it_d)
        writer.add_scalar('loss_c_kl', loss_c_info[1], global_step=it_d)
        writer.add_scalar('loss_c_cosine', loss_c_info[2], global_step=it_d)
        writer.add_scalar('loss_c_ssim', loss_c_info[3], global_step=it_d)
        writer.add_scalar('loss_c_lpips', loss_c_info[4], global_step=it_d)

        writer.add_scalars('Image_Space_MSE', {'loss_mask_mse':loss_mask_info[0][0],'loss_grad_mse':loss_grad_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_mask_kl':loss_mask_info[1],'loss_grad_kl':loss_grad_info[1],'loss_imgs_kl':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_mask_cosine':loss_mask_info[2],'loss_grad_cosine':loss_grad_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_mask_ssim':loss_mask_info[3],'loss_grad_ssim':loss_grad_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_mask_lpips':loss_mask_info[4],'loss_grad_lpips':loss_grad_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c_mse':loss_c_info[0][0],'loss_c_mse_mean':loss_c_info[0][1],'loss_c_mse_std':loss_c_info[0][2],'loss_c_kl':loss_w_info[1],'loss_c_cosine':loss_w_info[2]}, global_step=it_d)

        if epoch % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.png'%(epoch),nrow=n_row) # nrow=3
            heatmap=torch.cat((heatmap_1,heatmap_2))
            cam=torch.cat((cam_1,cam_2))
            grads = torch.cat((grad_1,grad_2))
            grads = grads.data.cpu().numpy() # [n,c,h,w]
            grads -= np.max(np.min(grads), 0)
            grads /= np.max(grads)
            torchvision.utils.save_image(torch.tensor(heatmap),resultPath_grad_cam+'/heatmap_%d.png'%(epoch),nrow=n_row)
            torchvision.utils.save_image(torch.tensor(cam),resultPath_grad_cam+'/cam_%d.png'%(epoch),nrow=n_row)
            torchvision.utils.save_image(torch.tensor(grads),resultPath_grad_cam+'/gb_%d.png'%(epoch),nrow=n_row)
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(epoch),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------ImageSpace--------',file=f)
                print('loss_mask_info: %s'%loss_mask_info,file=f)
                print('loss_grad_info: %s'%loss_grad_info,file=f)
                print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                print('loss_Gcam_info: %s'%loss_Gcam_info,file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
                print('loss_c_info: %s'%loss_c_info,file=f)
                #print('loss_condVector_info: %s'%loss_condVector_info,file=f)
            if epoch % 5000 == 0:
                torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)

if __name__ == "__main__":

    if not os.path.exists('./result'): os.mkdir('./result')

    resultPath = "./result/BigGAN256_attentionV2_argMax2OneHot_CBN_condVector_fixlabel_v2"
    if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/models"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    resultPath_grad_cam = resultPath+"/grad_cam"
    if not os.path.exists(resultPath_grad_cam): os.mkdir(resultPath_grad_cam)

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    cache_path = './checkpoint/biggan/256/G-256.pt'
    resolved_config_file = './checkpoint/biggan/256/biggan-deep-256-config.json'
    config = BigGANConfig.from_json_file(resolved_config_file)
    model = BigGAN(config)
    model.load_state_dict(torch.load(cache_path))
    model.eval()


    train(generator=model, tensor_writer=writer, synthesis_kwargs=0.4)