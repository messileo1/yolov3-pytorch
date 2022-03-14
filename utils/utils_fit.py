import torch
from tqdm import tqdm

from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):

    loss        = 0 #训练
    val_loss    = 0 #验证

    model_train.train() #在网络中含有BN层、dropout层时必须指明处于训练还是验证模式。因为BN和dropout在训练和测试时作用不同。具体百度
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar: #进度条，参数自行百度
        for iteration, batch in enumerate(gen): #gen和gen_val是一个dataloader，dataloader会一批一批的返回数据。
            if iteration >= epoch_step: #interation=总的样本数/batch_size，因此这个条件判断其实和 for i in range(训练轮数)是一样的作用
                break

            images, targets = batch[0], batch[1] #也可以写成images,targets = batch，batch是训练数据，会返回图片和标签。注意这里是一个batch的数据而不是单张图片的数据
            with torch.no_grad():# 作用是不累计梯度，保证不会进行调优。看代码层次，这一步不是对训练过程进行限定，而是保证数据在转换形式的时候不累计梯度
                if cuda: # Creates a Tensor from a numpy.ndarray.
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度，因为梯度在反向传播过程中是累加的，这意味着每次进行反向传播都会累加之前的梯度，所以反向传播之前必须进行梯度清零。
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            loss_value_all  = 0
            num_pos_all     = 0
            #----------------------#
            #   计算损失函数。一共循环3次，因为有3个输出特征层
            #----------------------#
            for l in range(len(outputs)):
                loss_item, num_pos = yolo_loss(l, outputs[l], targets) #l代表当前进入计算损失函数的是第几个特征层
                loss_value_all  += loss_item #三个特征层上，所有正样本的损失之和
                num_pos_all     += num_pos #三个特征层上，所有正样本的数量之和
            loss_value = loss_value_all / num_pos_all #取了平均。当然loss有不同计算方式，甚至可以再在loss_value基础上再除以100

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward() #根据loss进行一次反向传播
            optimizer.step() #更新权重

            loss += loss_value.item() #每个batch的loss累加，最后得到一个epoch的loss。随着训练轮数的增加这个loss应该呈现总体下降的趋势。
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)#在迭代程序每一步完成的时候更新进度条

    print('Finish Train')

    model_train.eval() # 置模型于验证/测试模式
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():  # 作用是不累计梯度，保证不会进行调优。
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item() # 验证损失
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
    #以字典的形式保存模型（推荐），加载的时候先定义模型结构，再加载模型参数。
    #此处保存的是整个模型的权重，当然可以修改传入的model参数为backbone来保存主干网络的权重
    #每训练一轮都将模型进行保存，当然可以加上条件判断，保存某轮之后的权重
