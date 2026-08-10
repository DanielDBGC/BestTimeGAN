import torch
import torch.nn as nn


def train_classifier(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    orig_labels_map=None,
    logger=None
):
    """
    Train a simple EEG classifier.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader or None
    epochs : int
    lr : float
    device : torch.device
    logger : optional logger
    """

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Build a reverse map: original label value -> local 0-based index
    # used by G and D (which have num_classes = n_gan_classes, not NUM_CLASSES).
    # E.g. if orig_labels_map = [2, 3], then raw label 2 -> 0, raw label 3 -> 1.
    # ------------------------------------------------------------------
    if orig_labels_map is not None:
        _inv_map = {int(v): i for i, v in enumerate(orig_labels_map.tolist())}
        def _to_local(raw_labels: torch.Tensor) -> torch.Tensor:
            """Remap raw dataset labels to 0-based local indices for Classifier."""
            return torch.tensor(
                [_inv_map[int(l)] for l in raw_labels.tolist()],
                dtype=torch.long,
                device=raw_labels.device,
            )
    else:
        def _to_local(raw_labels):
            return raw_labels

    for epoch in range(epochs):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:

            y = _to_local(y)

            X = X.to(device)
            y = y.to(device)

            
            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # -------------------------
        # Validation
        # -------------------------
        val_loss = None
        val_acc = None

        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad():
                for X, y in val_loader:

                    y = _to_local(y)

                    X = X.to(device)
                    y = y.to(device)

                    logits = model(X)
                    loss = criterion(logits, y)

                    val_loss += loss.item() * X.size(0)

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_loss /= total
            val_acc = correct / total

        # -------------------------
        # Logging
        # -------------------------
        if logger:
            if val_loader:
                logger.info(
                    f"Epoch {epoch+1} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                )
        else:
            if val_loader:
                print(
                    f"Epoch {epoch+1} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                )

    return model
