import time

import torch


def measure_latency(model, input_tensor, num_iterations=100):
    # Warm-up run
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_iterations
    return avg_latency * 1000  # Convert to milliseconds


if __name__ == "__main__":
    from models import SegmentationNetwork, reparameterize_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegmentationNetwork(
        layers=[2, 2, 4, 2],
        embed_dims=[48, 96, 192, 384],
        mlp_ratios=[3, 3, 3, 3],
        downsamples=[True, True, True, True],
        head_dim=64,
    )
    model.eval()

    # Reparameterize the model
    model = reparameterize_model(model)

    model.to(device)

    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 768, 1024).to(device)

    backbone_params = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

    print(f"Backbone parameters: {backbone_params / 1e6:.4}M")
    print(f"Head parameters: {head_params / 1e6:.4}M")
    print(f"Total parameters: {(backbone_params + head_params) / 1e6:.4}M\n")

    # Measure latency
    latency = measure_latency(model, input_tensor)
    print(f"Average latency: {latency:.2f} ms")
