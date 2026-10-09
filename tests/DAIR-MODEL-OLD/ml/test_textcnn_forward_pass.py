import torch

from ml.src.models.textcnn import TextCNN


def test_textcnn_forward_shape():
    model = TextCNN(
        vocab_size=100,
        embedding_dim=16,
        num_filters=8,
        kernel_sizes=[3, 4, 5],
        num_classes=6,
        dropout=0.3,
    )

    input_ids = torch.randint(
        low=0,
        high=100,
        size=(4, 64),
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape == (4, 6)


def test_textcnn_loss_is_finite():
    model = TextCNN(
        vocab_size=100,
        embedding_dim=16,
        num_filters=8,
        kernel_sizes=[3, 4, 5],
        num_classes=6,
        dropout=0.3,
    )

    input_ids = torch.randint(
        low=0,
        high=100,
        size=(4, 64),
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    logits = model(input_ids)

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    assert torch.isfinite(loss)


def test_textcnn_backward_produces_gradients():
    model = TextCNN(
        vocab_size=100,
        embedding_dim=16,
        num_filters=8,
        kernel_sizes=[3, 4, 5],
        num_classes=6,
        dropout=0.0,
    )

    input_ids = torch.randint(
        low=0,
        high=100,
        size=(4, 64),
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    logits = model(input_ids)

    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
    )

    loss.backward()

    assert model.embedding.weight.grad is not None

    assert model.classifier.weight.grad is not None

    for convolution in model.convolutions:
        assert convolution.weight.grad is not None
