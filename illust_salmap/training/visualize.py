from io import BytesIO

import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def generate_plot(title: str, images: dict[str, Tensor], figsize=(11, 8), dpi=350):
    """
    Renders the images side by side and returns the result as a PIL image.

    Built on Agg directly rather than through pyplot. The plot is only ever written to a
    buffer, but pyplot would still route it through whatever interactive backend is
    configured -- which opens a GUI window mid-training, and fails outright on a headless
    box or a broken Tk install. Going through the figure API also keeps pyplot's global
    figure registry out of it, so nothing accumulates across epochs.
    """
    figure = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(figure)

    figure.suptitle(title)

    # squeeze=False: a single image would otherwise yield a bare Axes instead of a row.
    axes = figure.subplots(1, len(images), squeeze=False)[0]

    for ax, (name, img) in zip(axes, images.items()):
        ax.set_title(name)
        ax.set_axis_off()
        ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy())

    figure.tight_layout()

    with BytesIO() as buffer:
        figure.savefig(buffer, format="png")
        # convert() forces the load before the buffer goes away.
        return Image.open(buffer).convert("RGB")


if __name__ == '__main__':
    writer = SummaryWriter("./tmp")

    for i in range(10):
        dummy_image = torch.rand(3, 32, 32)

        images = {
            "image": dummy_image, "saliency_map": dummy_image, "ground_truth": dummy_image,
        }

        image = generate_plot("test", images)
        writer.add_image("test", image, i)
        writer.flush()
