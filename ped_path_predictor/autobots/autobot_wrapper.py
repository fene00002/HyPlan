import os
import sys
import time

from matplotlib import pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

# Add all directories up to the current working directory to the Python path
current_dir = os.getcwd()
while current_dir:
    sys.path.append(current_dir)
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:  # Root directory reached
        break
    current_dir = parent_dir

from utils.logger import log_info, log_exception
from ped_path_predictor.autobots.new_util import singleDatasets, getDataloaders
from ped_path_predictor.autobots.autobot_ego import AutoBotEgo
from ped_path_predictor.autobots.train_helpers import nll_loss_multimodes


class AutoBotWrapperNew():

    def __init__(self, data_file_path=None, model_file_path=None):

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.n_obs = 15
        self.n_pred = 20
        self.k_attr = 2  # number of attributes per agent (x,y)
        self.batch_size = 512
    
        self.model = AutoBotEgo(
            k_attr=self.k_attr,
            d_k=128,
            _M=1,
            c=1,
            T=self.n_pred,
            L_enc=1,
            dropout=0.0,
            num_heads=16,
            L_dec=1,
            tx_hidden_size=384,
            use_map_img=False,
            use_map_lanes=False
        ).cuda()

        self.optimiser = optim.Adam(self.model.parameters(), lr=0.001, eps=1e-4)
        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5, verbose=True)

        if model_file_path is not None:
            self.model.load_state_dict(torch.load(model_file_path))
            log_info(f"using pedestrian path prediction with n_obs: {self.n_obs}, n_pred: {self.n_pred} and model: {self.model}")

        if data_file_path is not None:
            self.train_loader, self.test_loader, self.val_loader = getDataloaders(
                data_file_path, self.n_obs, self.n_pred, batch_size=self.batch_size, absolute=False
            )


    def transform(self, x, ego_out=None):
        if np.shape(x)[-1] == 6:
            ego_in = x[:, :, :2]
            agent_in = x[:, :, 2:]
        elif np.shape(x)[-1] == 4:
            ego_in = x[:, :, :2]
            agent_in = x[:, :, 2:] 
        else:
            raise ValueError(f"x must have shape (batch_size, n_obs, 4) or (batch_size, n_obs, 6), got {x.shape}")
        if ego_out is not None: ego_out = ego_out.cuda()

        map_lanes = torch.zeros((self.batch_size, 1, 1)).cuda()
        ex_mask = torch.ones((ego_in.shape[0], ego_in.shape[1], 1)).float()
        ego_in = torch.concatenate((ego_in, ex_mask), dim=-1).cuda()
        agents_in = torch.concatenate((agent_in, ex_mask), dim=-1).unsqueeze(-2).cuda()

        return ego_in, agents_in, map_lanes, ego_out


    def get_single_prediction(self, ped_path, ego_path):
        with torch.no_grad():
            if np.shape(ped_path)[0] != self.n_obs or np.shape(ego_path)[0] != self.n_obs:
                raise ValueError(f"ped_path and ego_path must have shape (n_obs, 2), got {ped_path.shape} and {ego_path.shape}")
            if np.shape(ped_path)[-1] != self.k_attr or np.shape(ego_path)[-1] != self.k_attr:
                raise ValueError(f"ped_path and ego_path must have shape (n_frames, 2), got {ped_path.shape} and {ego_path.shape}")

            x = np.concatenate((ped_path, ego_path), axis=-1, dtype=np.float32)
            x = torch.from_numpy(x).unsqueeze(0)  # shape (1, n_obs, 4)
            ego_in, agents_in, map_lanes, _  = self.transform(x)
            pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)
            pred_obs = pred_obs.squeeze()[:, :2].cpu().tolist()  # shape (n_pred, 2)
            return pred_obs


    def _compute_ego_errors(self, ego_preds, ego_gt, ego_in=None):
        with torch.no_grad():
            ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
            ade_losses = torch.mean(
                torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1
            ).transpose(0,1).cpu().numpy()

            fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1)\
                              .transpose(0,1).cpu().numpy()

            a = torch.square(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2]).sum(-1).sqrt().sum().item()
            f = torch.square((ego_preds[:, -1:, :, :2] - ego_gt[:, -1:, :, :2])).sum(-1).sqrt().sum().item()

            some = False
            if some == True:
                index = 435
                # # make output relative to the last observed frame
                i_t = ego_in[:, 60 - 1:, 0:2].detach().cpu().numpy()
                i_t = np.expand_dims(i_t, axis=1)
                i_t = np.repeat(i_t, 80, axis=1)
                i_t = i_t.squeeze(2)

                ego_gt = ego_gt.squeeze(0).transpose(0, 1)
                ego_gt = ego_gt[:, :, :2].cpu().numpy() + i_t

                ego_preds = ego_preds.squeeze(0).transpose(0, 1)
                ego_preds = ego_preds[:, :, :2].cpu().numpy() + i_t

                plt.plot(ego_in[index, :, 0].cpu().numpy(), ego_in[index, :, 1].cpu().numpy())
                plt.plot(ego_gt[index, :, 0], ego_gt[index, :, 1])
                plt.plot(ego_preds[index, :, 0], ego_preds[index, :, 1])

                plt.xlim(70, 100)
                plt.ylim(220, 250)

                plt.title("Sample Trajectory Prediction (Interactive 1)\n AutoBot")
                plt.legend(["Observed", "Ground Truth", "Predicted"])
                plt.xlabel("X-Coordinate")
                plt.ylabel("Y-Coordinate")
                plt.savefig("./pics/AutoBot_int2_435.svg", dpi=300)
                plt.savefig("./pics/AutoBot_int2_435.png", dpi=300)
                plt.show()

        return ade_losses, fde_losses, a, f


    def train(self):
        best_eval = np.Inf
        best_eval_fde = np.Inf
        last_best_epoch = 0

        for epoch in range(0, 1000):
            print(f'Epoch {epoch}')
            did_epoch_better = False

            self.model.train()

            t_before = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                print(f'\rBatch {i}/{len(self.train_loader)}')

                ego_in, agents_in, map_lanes, ego_out = self.transform(x, y)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)

                nll_loss, kl_loss, post_entropy, ade_fde_loss = nll_loss_multimodes(
                    pred_obs, ego_out[:, :, :2],
                    mode_probs,
                    entropy_weight=40.0,
                    kl_weight=20.0,
                    use_FDEADE_aux_loss=True
                )

                self.optimiser.zero_grad()
                (nll_loss + ade_fde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()

                if i % 100 == 0:
                    print(f"Epoch: {epoch:4}, Batch: {i:4} Loss: {ade_fde_loss:.6f} Time: {(time.time() - t_before): 4.4f}")
                    t_before = time.time()

                    eval_loss, fde_loss = self.eval(self.val_loader)

                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                        did_epoch_better = True
                        print(f"Saving Model with loss:{eval_loss:.4f},{fde_loss:.4f}")
                        torch.save(self.model.state_dict(), self.models_dir + f"/autobots_ego_{epoch}.pth")
                    self.model.train()

            if did_epoch_better:
                print(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
                last_best_epoch = epoch
            if epoch - last_best_epoch > 50:
                print(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
                break
            self.optimiser_scheduler.step()


    def eval(self, dataloader):
        eval_loss = 0
        fde_loss = 0
        self.model.eval()
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(dataloader):
                print(f'\rBatch {j}/{len(dataloader)}', end='')
                ego_in, agents_in, map_lanes, ego_out = self.transform(x_val, y_val)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)

                ade_losses, fde_losses, a, f = self._compute_ego_errors(pred_obs, ego_out, ego_in=ego_in)

                eval_loss += a / self.n_pred
                fde_loss += f
        self.model.train()
        eval_loss /= len(dataloader) * self.batch_size
        fde_loss /= len(dataloader) * self.batch_size
        return eval_loss, fde_loss


if __name__ == '__main__':
    # Use the absolute path of the current script to construct the file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(base_dir, "data/pedestrian_path_data_26.06.2025_19.12.39.json")
    model_file_path = os.path.join(base_dir, "autobot_ego.pth")

    if "--train" in sys.argv:
        abw = AutoBotWrapperNew(data_file_path=data_file_path)
        abw.train()
    elif "--val" in sys.argv:
        raise NotImplementedError("Validation mode is not implemented yet.")
    elif "--test" in sys.argv:
        abw = AutoBotWrapperNew(data_file_path=data_file_path, model_file_path=model_file_path)
        eval_loss, fde_loss = abw.eval(abw.test_loader)
        print(f"Eval Loss: {eval_loss}, FDE Loss: {fde_loss}")
