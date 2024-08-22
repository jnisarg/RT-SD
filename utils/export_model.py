import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.onnx


def create_dummy_input(input_shape, device="cuda"):
    return torch.randn(input_shape).to(device)


def simplify_onnx(onnx_model_path, simplified_model_path):
    onnx_model = onnx.load(onnx_model_path)
    model_simp, check = onnxsim.simplify(onnx_model)

    if check:
        onnx.save(model_simp, simplified_model_path)
        print(f"Simplified ONNX model saved to {simplified_model_path}")
    else:
        print("Failed to simplify ONNX model")


def export_to_onnx(
    model, dummy_input, onnx_path, input_names, output_names, dynamic_axes=None
):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"ONNX model exported to {onnx_path}")


def optimize_and_export_model(
    model,
    input_shape,
    onnx_path,
    simplified_onnx_path,
    input_names,
    output_names,
    dynamic_axes=None,
    device="cuda",
):
    model.eval()
    model.to(device)

    dummy_input = create_dummy_input(input_shape, device)

    # Export to ONNX
    export_to_onnx(
        model, dummy_input, onnx_path, input_names, output_names, dynamic_axes
    )

    # Simplify ONNX model
    simplify_onnx(onnx_path, simplified_onnx_path)


# Example usage
if __name__ == "__main__":

    from models import SegmentationNetwork, reparameterize_model

    model = SegmentationNetwork(
        layers=[2, 2, 4, 2],
        embed_dims=[48, 96, 192, 384],
        mlp_ratios=[3, 3, 3, 3],
        downsamples=[True, True, True, True],
        head_dim=64,
        use_layer_scale=False,
    )

    # Reparameterize the model
    model = reparameterize_model(model)

    class SimpleModel(nn.Module):
        def __init__(self, model):
            super(SimpleModel, self).__init__()
            self.model = model

        def forward(self, x):
            return torch.argmax(self.model(x), dim=1, keepdim=True)

    model = SimpleModel(model)
    model.eval()

    # Set parameters
    input_shape = (1, 3, 768, 1024)
    onnx_path = "assets/model.onnx"
    simplified_onnx_path = "assets/model_simplified.onnx"
    input_names = ["input"]
    output_names = ["output"]
    # dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Optimize and export the model
    optimize_and_export_model(
        model,
        input_shape,
        onnx_path,
        simplified_onnx_path,
        input_names,
        output_names,
        # dynamic_axes,
    )
