import torch
import model.stylegan2_generator as model
import torchvision
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

generator = model.StyleGAN2Generator(resolution=1024) #包含了三个类，Gm, Gs, Trunck(这个类没用参数)

#加载
checkpoint = torch.load('/Users/apple/Desktop/genforce/checkpoint/stylegan2_ffhq1024.pth', map_location='cpu')
if 'generator_smooth' in checkpoint: #默认是这个
    generator.load_state_dict(checkpoint['generator_smooth'])
else:
    generator.load_state_dict(checkpoint['generator'])

# ## 三个模型总览
z = torch.randn(1,512)
# # output = generator(z)
# # print(output.keys())#输出是一个字典
# # print(output['w'].shape) # [n,512]
# # print(output['wp'].shape) # [n,18,512]
# # print(output['image'].shape)

# ## Gm, 注意这里的w只有1维，不是18维
# # Gm = generator.mapping
# # output_gm = Gm(z)
# # print(output_gm.keys())

# ## Gs
# wp=torch.randn(3,18,512)#Wp [n,18,512]
trunc_=5
synthesis_kwargs = dict(trunc_psi=0.7,trunc_layers=trunc_,randomize_noise=False) #  6:256, 7:512, 8:1024
Gs = generator.synthesis
#images = Gs(wp)['image']
#print(Gs.early_layer.const.shape)

# ## Gtrunc
Gt = generator.truncation
# with torch.no_grad():
#     w = torch.randn(5,18,512)
#     w2 = Gt(w,trunc_psi=0.7,trunc_layers=8)
#     print(w2[0,9]==w[0,9])


#--Const [n,512,4,4]
# w = torch.randn(1)
# const_input = Gs.early_layer(w)
# print(const_input.shape) # [1,512,4,4]

#z[0,511]=100
with torch.no_grad():
    #images = generator(z, **synthesis_kwargs)['image']
    result_dict = generator(z, **synthesis_kwargs)
    wp = result_dict['wp']
    #wp[0,10,180]=80
    img1 = result_dict['image']
    result_dict2 = Gs(wp)
    print(result_dict2.keys())
    img2 = result_dict2['image']
torchvision.utils.save_image(img1*0.5+0.5,'./img1.png',nrow=1)
torchvision.utils.save_image(img2*0.5+0.5,'./img2.png',nrow=1)



