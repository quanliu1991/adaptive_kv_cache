import os.path

import torch
import matplotlib.pyplot as plt


def image_scores(scores: torch.Tensor, layer_id):
    bs, num_heads, q_seq_len, k_seq_len = map(int, scores.shape)

    rows = 4
    cols = int(num_heads/rows)

    fig, axs = plt.subplots(rows, cols, figsize=(30, 10))


    for i in range(32):
        slice_tensor = scores[0, i, :, :]
        slice_np = slice_tensor.cpu().numpy()


        axs[i%rows][i//rows].imshow(slice_np, cmap='hot', interpolation='nearest')
        axs[i%rows][i//rows].set_title(f'h{i+1}')
        axs[i%rows][i//rows].axis('off')

    plt.tight_layout()
    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"images")
    os.makedirs(image_path,exist_ok=True)

    plt.savefig(os.path.join(image_path,f'layer{layer_id}_attention.png'))


