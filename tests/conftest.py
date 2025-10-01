"""Pytest configuration and shared fixtures for oscillatory test suite."""

import pytest
import torch


@pytest.fixture(params=["cpu"])
def device(request):
    """Provide device for testing (CPU always, CUDA if available)."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    """Provide dtype for testing."""
    return request.param


@pytest.fixture
def seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def reset_seed(seed):
    """Reset random seed before each test."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def image_size():
    """Default image size for tests."""
    return 32


@pytest.fixture
def n_channels():
    """Default number of channels."""
    return 3


@pytest.fixture
def sample_image(batch_size, n_channels, image_size, device, dtype):
    """Create a sample image tensor."""
    return torch.randn(
        batch_size, n_channels, image_size, image_size, device=device, dtype=dtype
    )


@pytest.fixture
def sample_angles(batch_size, n_channels, image_size, device, dtype):
    """Create sample angle tensor in [-π, π]."""
    return (
        torch.randn(
            batch_size, n_channels, image_size, image_size, device=device, dtype=dtype
        )
        * 3.14
    )


class GradientChecker:
    """Helper class for checking gradients."""

    @staticmethod
    def check_gradients(model, inputs, outputs=None):
        """Check that gradients flow through model."""
        if outputs is None:
            outputs = (
                model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
            )

        loss: torch.Tensor
        if isinstance(outputs, (list, tuple)):
            loss = sum(
                o.sum() if o.numel() > 1 else o
                for o in outputs
                if isinstance(o, torch.Tensor)
            )  # type: ignore[assignment]
        else:
            loss = outputs.sum() if outputs.numel() > 1 else outputs

        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient not computed for parameter"
                has_grad = True

        assert has_grad, "No gradients found in model"

    @staticmethod
    def numerical_gradient_check(func, inputs, eps=1e-5, rtol=1e-3, atol=1e-4):
        """Check analytical gradients against numerical gradients."""
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [
            inp.detach().requires_grad_(True) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]

        # Forward pass
        output = func(*inputs)
        if isinstance(output, (list, tuple)):
            output = output[0]

        # Analytical gradient
        output.sum().backward()
        analytical_grads = [
            inp.grad.clone() if inp.grad is not None else None
            for inp in inputs
            if isinstance(inp, torch.Tensor)
        ]

        # Numerical gradient
        numerical_grads = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor) or not inp.requires_grad:
                continue

            grad = torch.zeros_like(inp)
            for idx in torch.ndindex(inp.shape):  # type: ignore[attr-defined]
                # Forward difference
                inp.data[idx] += eps
                out_plus = func(*inputs)
                if isinstance(out_plus, (list, tuple)):
                    out_plus = out_plus[0]

                inp.data[idx] -= 2 * eps
                out_minus = func(*inputs)
                if isinstance(out_minus, (list, tuple)):
                    out_minus = out_minus[0]

                grad[idx] = (out_plus.sum() - out_minus.sum()) / (2 * eps)

                # Restore original value
                inp.data[idx] += eps

            numerical_grads.append(grad)

        # Compare
        for analytical, numerical in zip(analytical_grads, numerical_grads):
            if analytical is not None and numerical is not None:
                assert torch.allclose(analytical, numerical, rtol=rtol, atol=atol), (
                    f"Gradient mismatch: max diff = {(analytical - numerical).abs().max()}"
                )


@pytest.fixture
def gradient_checker():
    """Provide gradient checking utility."""
    return GradientChecker()


@pytest.fixture
def performance_threshold():
    """Performance thresholds for regression testing."""
    return {
        "memory_increase_factor": 1.1,  # Allow 10% memory increase
        "time_increase_factor": 1.2,  # Allow 20% time increase
    }


def assert_shape(tensor, expected_shape, msg=""):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, (
        f"{msg}Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_device(tensor, expected_device, msg=""):
    """Assert tensor is on expected device."""
    assert tensor.device == expected_device, (
        f"{msg}Expected device {expected_device}, got {tensor.device}"
    )


def assert_dtype(tensor, expected_dtype, msg=""):
    """Assert tensor has expected dtype."""
    assert tensor.dtype == expected_dtype, (
        f"{msg}Expected dtype {expected_dtype}, got {tensor.dtype}"
    )


def assert_finite(tensor, msg=""):
    """Assert all tensor values are finite."""
    assert torch.isfinite(tensor).all(), f"{msg}Tensor contains non-finite values"


def assert_unit_norm(tensor, dim, eps=1e-6, msg=""):
    """Assert tensor has unit norm along dimension."""
    norms = tensor.norm(dim=dim, p=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=eps), (
        f"{msg}Tensor not unit normalized along dim {dim}: norm range [{norms.min()}, {norms.max()}]"
    )
