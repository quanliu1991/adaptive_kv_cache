import torch
import matplotlib.pyplot as plt


def image_scores(scores: torch.Tensor):
    bs, num_heads, seq_len,seq_len = map(int, scores.shape)

    rows = bs
    cols = num_heads

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5))

    for i in range(rows):
        for j in range(cols):
            slice_tensor = scores[i, j, :, :]
            slice_np = slice_tensor.cpu().numpy()


            axs[i][j].imshow(slice_np, cmap='hot', interpolation='nearest')
            axs[i][j].set_title(f'b {i + 1} h {j+1}')
            axs[i][j].axis('off')

    plt.tight_layout()
    plt.savefig('attention_heatmap.png')
    plt.show()
