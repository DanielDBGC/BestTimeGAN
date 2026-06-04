import torch
import torch.nn as nn


def train_classifier(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
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

    for epoch in range(epochs):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
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
