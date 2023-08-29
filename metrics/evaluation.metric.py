import torch
from tqdm import tqdm
from monai.data import decollate_batch


class ValidationEvaluator:
    @staticmethod
    @torch.inference_mode()
    def validate(model, dataloader, metric, metric_batch, post_trans, device, amp, VAE_param=False):
        model.eval()
        num_val_batches = len(dataloader)
        dice_score = 0

        # iterate over the validation set
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_val_batches, dynamic_ncols=True,
                              desc='Validation round', unit='batch', leave=False):
                val_inputs, val_labels = (
                    batch["images"].to(device),
                    batch["mask"].to(device),
                )

                if VAE_param:
                    val_outputs, loss = model(val_inputs)
                else:
                    val_outputs = model(val_inputs)

                val_outputs = [post_trans(i)
                               for i in decollate_batch(val_outputs)]
                metric(y_pred=val_outputs, y=val_labels)
                metric_batch(y_pred=val_outputs, y=val_labels)

        model.train()
