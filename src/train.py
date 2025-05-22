import torch
from torch.optim import AdamW
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time

# # 定义训练主循环类
# args	所有超参数
# writer	TensorBoard 用于记录训练指标
# model	UniST 模型
# data / val_data / test_data	三份数据：train / val / test
# device	GPU / CPU
# self.opt	优化器：AdamW
# self.mask_list	不同 mask 策略及默认掩盖比例
class TrainLoop:
    def __init__(self, args, writer, model, data, test_data, val_data, device, early_stop=5):
        # 初始化基本参数
        self.args = args
        self.writer = writer
        self.model = model
        self.data = data
        self.test_data = test_data
        self.val_data = val_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        # 只优化需要更新的参数
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=self.weight_decay)
        self.log_interval = args.log_interval
        self.best_rmse_random = 1e9  # 随机掩盖测试最优rmse初始化
        self.warmup_steps = 5  # 学习率预热步数
        self.min_lr = args.min_lr  # 最小学习率
        self.best_rmse = 1e9  # 验证集上最佳RMSE初始化
        self.early_stop = early_stop  # 早停容忍次数
        # 定义支持的masking策略及mask ratio
        self.mask_list = {'random':[0.5],'temporal':[0.5],'tube':[0.5],'block':[0.5]}

    # 单步训练
    def run_step(self, batch, step, mask_ratio, mask_strategy, index, name):
        self.opt.zero_grad()  # 梯度清零
        # 前向 + 反向传播，计算loss
        loss, num, loss_real, num2 = self.forward_backward(batch, step, mask_ratio, mask_strategy, index=index, name=name)
        self._anneal_lr()  # 动态调整学习率
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)  # 梯度裁剪
        self.opt.step()  # 更新参数
        return loss, num, loss_real, num2

    # 测试阶段采样
    def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0, Type='val'):
        with torch.no_grad():  # 禁止梯度计算，加速推理
            error_mae, error_norm, error, num, error2, num2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for _, batch in enumerate(test_data[index]):
                # 前向推理，得到预测
                loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, seed=seed, data=dataset, mode='forward')
                pred = torch.clamp(pred, min=-1, max=1)  # 限制输出范围
                pred_mask = pred.squeeze(dim=2)
                target_mask = target.squeeze(dim=2)
                # 计算评价指标（RMSE，MAE）
                error += mean_squared_error(
                    self.args.scaler[dataset].inverse_transform(pred_mask[mask==1].reshape(-1,1).detach().cpu().numpy()),
                    self.args.scaler[dataset].inverse_transform(target_mask[mask==1].reshape(-1,1).detach().cpu().numpy()),
                    squared=True
                ) * mask.sum().item()
                error_mae += mean_absolute_error(
                    self.args.scaler[dataset].inverse_transform(pred_mask[mask==1].reshape(-1,1).detach().cpu().numpy()),
                    self.args.scaler[dataset].inverse_transform(target_mask[mask==1].reshape(-1,1).detach().cpu().numpy())
                ) * mask.sum().item()
                error_norm += loss.item() * mask.sum().item()
                num += mask.sum().item()
                num2 += (1-mask).sum().item()

        rmse = np.sqrt(error / num)  # 均方根误差
        mae = error_mae / num  # 平均绝对误差
        loss_test = error_norm / num  # 测试loss
        return rmse, mae, loss_test

    # 评估函数，验证集或测试集使用
    def Evaluation(self, test_data, epoch, seed=None, best=True, Type='val'):
        loss_list = []
        rmse_list = []
        rmse_key_result = {}

        for index, dataset_name in enumerate(self.args.dataset.split('*')):
            rmse_key_result[dataset_name] = {}
            if self.args.mask_strategy_random != 'none':
                for s in self.mask_list:
                    for m in self.mask_list[s]:
                        # 遍历所有mask策略和mask比例进行验证
                        result, mae, loss_test = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy=s, seed=seed, dataset=dataset_name, index=index, Type=Type)
                        rmse_list.append(result)
                        loss_list.append(loss_test)
                        if s not in rmse_key_result[dataset_name]:
                            rmse_key_result[dataset_name][s] = {}
                        rmse_key_result[dataset_name][s][m] = result
                        # 写入tensorboard
                        if Type == 'val':
                            self.writer.add_scalar('Evaluation/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                        elif Type == 'test':
                            self.writer.add_scalar('Test_RMSE/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                            self.writer.add_scalar('Test_MAE/MAE-{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), mae, epoch)

        loss_test = np.mean(loss_list)  # 平均测试loss

        if best:
            is_break = self.best_model_save(epoch, loss_test, rmse_key_result)
            return is_break
        else:
            return loss_test, rmse_key_result

    # 最优模型保存逻辑
    def best_model_save(self, step, rmse, rmse_key_result):
        if rmse < self.best_rmse:
            self.early_stop = 0  # 重置early stop计数
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_stage_{}.pkl'.format(self.args.stage))
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
            self.best_rmse = rmse
            self.writer.add_scalar('Evaluation/RMSE_best', self.best_rmse, step)
            print('\nRMSE_best:{}\n'.format(self.best_rmse))
            print(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result.txt', 'w') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            return 'save'
        else:
            self.early_stop += 1
            print('\nRMSE:{}, RMSE_best:{}, early_stop:{}\n'.format(rmse, self.best_rmse, self.early_stop))
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('RMSE:{}, not optimized, early_stop:{}\n'.format(rmse, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                with open(self.args.model_path+'result.txt', 'a') as f:
                    f.write('Early stop!\n')
                with open(self.args.model_path+'result_all.txt', 'a') as f:
                    f.write('Early stop!\n')
                exit()

    # 在random策略中，随机选择掩盖策略
    def mask_select(self):
        if self.args.mask_strategy_random == 'none':
            mask_strategy = self.args.mask_strategy
            mask_ratio = self.args.mask_ratio
        else:
            mask_strategy = random.choice(['random','temporal','tube','block'])
            mask_ratio = random.choice(self.mask_list[mask_strategy])
        return mask_strategy, mask_ratio

    # 总训练循环
    def run_loop(self):
        step = 0

        if self.args.mode == 'testing':
            self.Evaluation(self.val_data, 0, best=True, Type='val')
            exit()
        
        self.Evaluation(self.val_data, 0, best=True, Type='val')

        for epoch in range(self.args.total_epoches):
            print('Training')
            self.step = epoch
            loss_all, num_all, loss_real_all, num_all2 = 0.0, 0.0, 0.0, 0.0
            start = time.time()
            for name, batch in self.data:
                mask_strategy, mask_ratio = self.mask_select()
                loss, num, loss_real, num2 = self.run_step(batch, step, mask_ratio=mask_ratio, mask_strategy=mask_strategy, index=0, name=name)
                step += 1
                loss_all += loss * num
                loss_real_all += loss_real * num
                num_all += num
                num_all2 += num2
            end = time.time()
            print('training time:{} min'.format(round((end-start)/60.0,2)))
            print('epoch:{}, training loss:{}, training rmse:{}'.format(epoch, loss_all/num_all, np.sqrt(loss_real_all/num_all)))

            if epoch % self.log_interval == 0 and epoch > 0 or epoch == 10 or epoch == self.args.total_epoches-1:
                print('Evaluation')
                eval_result = self.Evaluation(self.val_data, epoch, best=True, Type='val')
                if eval_result == 'save':
                    print('test evaluate!')
                    rmse_test, rmse_key_test = self.Evaluation(self.test_data, epoch, best=False, Type='test')
                    print('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                    print(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result_all.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')

    # 前向推理（用于train/test）
    def model_forward(self, batch, model, mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):
        batch = [i.to(self.device) for i in batch]
        loss, loss2, pred, target, mask = self.model(
            batch,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            seed=seed,
            data=data,
            mode=mode,
        )
        return loss, loss2, pred, target, mask

    # 前向 + 反向传播
    def forward_backward(self, batch, step, mask_ratio, mask_strategy, index, name=None):
        loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, data=name, mode='backward')
        pred_mask = pred.squeeze(dim=2)[mask==1]
        target_mask = target.squeeze(dim=2)[mask==1]
        loss_real = mean_squared_error(
            self.args.scaler[name].inverse_transform(pred_mask.reshape(-1,1).detach().cpu().numpy()),
            self.args.scaler[name].inverse_transform(target_mask.reshape(-1,1).detach().cpu().numpy()),
            squared=True
        )
        loss.backward()
        self.writer.add_scalar('Training/Loss_step', np.sqrt(loss_real), step)
        return loss.item(), mask.sum().item(), loss_real, (1-mask).sum().item()

    # 动态调整学习率
    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
        elif self.step < self.lr_anneal_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (self.step - self.warmup_steps) / (self.lr_anneal_steps - self.warmup_steps))
            )
        else:
            lr = self.min_lr
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
