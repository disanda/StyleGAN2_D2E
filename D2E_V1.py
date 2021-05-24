#training StyleGANv2

import torch
import model.E_v3 as BE
import model.stylegan2_generator as model
from model.utils.custom_adam import LREQAdam
import metric.pytorch_ssim as pytorch_ssim
import os
import lpips
import numpy as np
import tensorboardX

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

    loss_imgs_mse_1 = loss_mse(imgs1,imgs2)
    loss_imgs_mse_2 = loss_mse(imgs1.mean(),imgs2.mean())
    loss_imgs_mse_3 = loss_mse(imgs1.std(),imgs2.std())
    loss_imgs_mse = loss_imgs_mse_1 + loss_imgs_mse_2 + loss_imgs_mse_3

    imgs1_kl, imgs2_kl = torch.nn.functional.softmax(imgs1),torch.nn.functional.softmax(imgs2)
    loss_imgs_kl = loss_kl(torch.log(imgs2_kl),imgs1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
    loss_imgs_kl = torch.where(torch.isnan(loss_imgs_kl),torch.full_like(loss_imgs_kl,0), loss_imgs_kl)
    loss_imgs_kl = torch.where(torch.isinf(loss_imgs_kl),torch.full_like(loss_imgs_kl,1), loss_imgs_kl)

    imgs1_cos = imgs1.contiguous.view(-1)
    imgs2_cos = imgs2.contiguous.view(-1)
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
    Gs = generator.synthesis
    Gm = generator.mapping
    E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
    #E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/EAE-car-cat/result/EB_cat_cosine_v2/E_model_ep80000.pth'))
    Gs.cuda()
    E.cuda()
    writer = tensor_writer

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0) 
    #用这个adam不会报错:RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

    batch_size = 3
    const_r = torch.randn(batch_size)
    const1 = Gs.early_layer(const_r) #[n,512,4,4]
    it_d = 0
    for epoch in range(0,250001):
        set_seed(epoch%30000)
        z = torch.randn(batch_size, 512).cuda() #[32, 512]
        with torch.no_grad(): #这里需要生成图片和变量
            result_all = generator(z, **synthesis_kwargs)
            imgs1 = result_all['image']
            w1 = result_all['wp']
        const2,w2 = E(imgs1.cuda())

        imgs2=Gs(w2)['image']

        E_optimizer.zero_grad()

#Latent_space

## c
        loss_c, loss_c_info = space_loss(const1,const2,image_space = False)
        E_optimizer.zero_grad()
        loss_c.backward(retain_graph=True)
        E_optimizer.step()

## w
        loss_w, loss_w_info = space_loss(w1,w2,image_space = False)
        E_optimizer.zero_grad()
        loss_w.backward(retain_graph=True)
        E_optimizer.step()


#Image Space 

##loss1 最小区域
        imgs_small_1 = imgs1[:,:,imgs1.shape[2]//20:-imgs1.shape[2]//20,imgs1.shape[3]//20:-imgs1.shape[3]//20].clone() # w,h
        imgs_small_2 = imgs2[:,:,imgs2.shape[2]//20:-imgs2.shape[2]//20,imgs2.shape[3]//20:-imgs2.shape[3]//20].clone()
        loss_small, loss_small_info = space_loss(imgs_small_1,imgs_small_2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        loss_small.backward(retain_graph=True)
        E_optimizer.step()


#loss2 中等区域
        imgs_medium_1 = imgs1[:,:,imgs1.shape[2]//10:-imgs1.shape[2]//10,imgs1.shape[3]//10:-imgs1.shape[3]//10].clone()
        imgs_medium_2 = imgs2[:,:,imgs2.shape[2]//10:-imgs2.shape[2]//10,imgs2.shape[3]//10:-imgs2.shape[3]//10].clone()
        loss_medium, loss_medium_info = space_loss(imgs_medium_1,imgs_medium_2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        loss_medium.backward(retain_graph=True)
        E_optimizer.step()

#loss3 原图区域
        loss_imgs, loss_imgs_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        loss_imgs.backward(retain_graph=True)
        E_optimizer.step()

        print('i_'+str(epoch))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_ssim, loss_imgs_cosine, loss_kl_imgs, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_small_info: %s'%loss_small_info)
        print('loss_medium_info: %s'%loss_medium_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c_info: %s'%loss_c_info)

        it_d += 1
        writer.add_scalar('loss_small_mse', loss_small_info[0][0], global_step=it_d)
        writer.add_scalar('loss_samll_mse_mean', loss_small_info[0][1], global_step=it_d)
        writer.add_scalar('loss_samll_mse_std', loss_small_info[0][2], global_step=it_d)
        writer.add_scalar('loss_samll_kl', loss_small_info[1], global_step=it_d)
        writer.add_scalar('loss_samll_cosine', loss_small_info[2], global_step=it_d)
        writer.add_scalar('loss_samll_ssim', loss_small_info[3], global_step=it_d)
        writer.add_scalar('loss_samll_lpips', loss_small_info[4], global_step=it_d)

        writer.add_scalar('loss_medium_mse', loss_medium_info[0][0], global_step=it_d)
        writer.add_scalar('loss_medium_mse_mean', loss_medium_info[0][1], global_step=it_d)
        writer.add_scalar('loss_medium_mse_std', loss_medium_info[0][2], global_step=it_d)
        writer.add_scalar('loss_medium_kl', loss_medium_info[1], global_step=it_d)
        writer.add_scalar('loss_medium_cosine', loss_medium_info[2], global_step=it_d)
        writer.add_scalar('loss_medium_ssim', loss_medium_info[3], global_step=it_d)
        writer.add_scalar('loss_medium_lpips', loss_medium_info[4], global_step=it_d)

        writer.add_scalar('loss_imgs_mse', loss_imgs_info[0][0], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_mean', loss_imgs_info[0][1], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_std', loss_imgs_info[0][2], global_step=it_d)
        writer.add_scalar('loss_imgs_kl', loss_imgs_info[1], global_step=it_d)
        writer.add_scalar('loss_imgs_cosine', loss_imgs_info[2], global_step=it_d)
        writer.add_scalar('loss_imgs_ssim', loss_imgs_info[3], global_step=it_d)
        writer.add_scalar('loss_imgs_lpips', loss_imgs_info[4], global_step=it_d)

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

        writer.add_scalars('Image_Space_MSE', {'loss_small_mse':loss_small_info[0][0],'loss_medium_mse':loss_medium_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_small_kl':loss_small_info[1],'loss_medium_kl':loss_medium_info[1],'loss_imgs_kl':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_samll_cosine':loss_small_info[2],'loss_medium_cosine':loss_medium_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_small_ssim':loss_small_info[3],'loss_medium_ssim':loss_medium_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_small_lpips':loss_small_info[4],'loss_medium_lpips':loss_medium_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c_mse':loss_c_info[0][0],'loss_c_mse_mean':loss_c_info[0][1],'loss_c_mse_std':loss_c_info[0][2],'loss_c_kl':loss_c_info[1],'loss_c_cosine':loss_w_info[2]}, global_step=it_d)


        if epoch % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.jpg'%(epoch),nrow=n_row) # nrow=3
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(epoch),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------ImageSpace--------',file=f)
                print('loss_small_info: %s'%loss_small_info,file=f)
                print('loss_medium_info: %s'%loss_medium_info,file=f)
                print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
                print('loss_c_info: %s'%loss_c_info,file=f)
            if epoch % 5000 == 0:
                torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)

if __name__ == "__main__":
    resultPath = "./result/StyleGANv2_horse256_attentionV1"
    if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/models"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    generator = model.StyleGAN2Generator(resolution=256).to(device)
    checkpoint = torch.load('./checkpoint/stylegan2_horse256.pth') #map_location='cpu'
    if 'generator_smooth' in checkpoint: #默认是这个
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])

    synthesis_kwargs = dict(trunc_psi=0.7,trunc_layers=8,randomize_noise=False)

    train(generator=generator, tensor_writer=writer, synthesis_kwargs=synthesis_kwargs )
