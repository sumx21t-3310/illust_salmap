from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torchvision.transforms.v2.functional import pil_to_tensor

from illust_salmap.training.losses import SaliencyLoss
from illust_salmap.training.metrics import SaliencyMetrics, normalized
from illust_salmap.training.visualize import generate_plot


def default_optimization_builder(params):
    return Adam(params, lr=0.0001)


class SaliencyModel(LightningModule):
    def __init__(
            self,
            model: Module,
            criterion: Module = None,
            optimization_builder: callable = default_optimization_builder, ):
        super().__init__()
        self.model = model
        # Built here rather than defaulted in the signature: a default argument is
        # evaluated once at import and then shared by every instance, which silently
        # shares parameters as soon as a criterion has any.
        self.criterion = criterion if criterion is not None else SaliencyLoss()
        self.optimization_builder = optimization_builder

        self.val_metrics = SaliencyMetrics("val_")
        self.test_metrics = SaliencyMetrics("test_")

    def forward(self, x) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return self.optimization_builder(self.parameters())

    def training_step(self, batch, batch_idx) -> dict[str, Tensor]:
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        return {"loss": loss, "predict": predict}

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        loss = outputs["loss"]

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)

    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        return {"val_loss": loss, "val_predict": predict}

    @torch.no_grad()
    def on_validation_batch_end(
            self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = outputs["val_loss"]
        predict = outputs["val_predict"]
        image, ground_truth = batch

        self.log("val_loss", loss, on_step=False, on_epoch=True, enable_graph=False)
        self.log_dict(self.val_metrics.update(predict, ground_truth),
                      on_step=False, on_epoch=True, enable_graph=False)

        if batch_idx == 0:
            self.save_image("validation", self.trainer.current_epoch, image, ground_truth, predict)

    def on_validation_epoch_end(self) -> None:
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        image, ground_truth = batch
        predict = self.forward(image)

        loss = self.criterion(predict, ground_truth)

        return {"test_loss": loss, "test_predict": predict}

    @torch.no_grad()
    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        loss = outputs["test_loss"]
        predict = outputs["test_predict"]
        image, ground_truth = batch

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)
        self.log_dict(self.test_metrics.update(predict, ground_truth),
                      on_step=False, on_epoch=True, prog_bar=True, enable_graph=False)

        if batch_idx == self.trainer.num_test_batches[0] - 1:
            self.save_image("test", self.trainer.current_epoch, image, ground_truth, predict)

    def on_test_epoch_end(self) -> None:
        torch.cuda.empty_cache()

    @torch.no_grad()
    def save_image(self, stage: str, epoch: int, images: Tensor, ground_truths: Tensor, predicts: Tensor) -> None:
        images = normalized(images)
        ground_truths = normalized(ground_truths)
        predicts = normalized(predicts)
        title = f"{stage}_images: {epoch}"

        plot = generate_plot(title, {"input": images[0], "ground_truth": ground_truths[0], "predict": predicts[0]})

        self.logger.experiment.add_image(f"{stage}_images", pil_to_tensor(plot), global_step=epoch)
