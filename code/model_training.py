import torch
import evaluate
import torch
from transformers import get_scheduler
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm


class ModelPipeline:
    def __init__(self, model_name, model_hyperparams, device):
        self.model_name = model_name
        self.model_hyperparams = model_hyperparams
        self.learning_rate = model_hyperparams.get('LEARNING_RATE')
        self.num_epochs = model_hyperparams.get('NUM_EPOCHS')
        self.device = device

    def __call__(self, num_labels=2):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                    num_labels=num_labels)
        self.optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        return model


    def model_training_loop(self, model, train_dataloader):
        #model = torch.nn.DataParallel(model)
        model.to(self.device)

        print(f"######### \ndevice\n#########\n {self.device}")
        progress_bar = tqdm(range(self.num_training_steps))
        model.train()
        for epoch in range(self.num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
        return model

    def train(self, model, train_dataloader):
        # Getting Training Hyperparams
        self.num_training_steps = self.num_epochs * len(train_dataloader)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps
        )
        model = self.model_training_loop(model, train_dataloader)
        return model

    def evaluate_model(self, tuned_model, eval_dataloader, MODEL_EVAL_PATH):
        evaluation_metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])
        tuned_model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = tuned_model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            evaluation_metrics.add_batch(
                predictions=predictions,
                references=batch["labels"]
            )
        results = evaluation_metrics.compute()

        hyperparams = {"model": self.model_name}
        hyperparams.update(self.model_hyperparams)

        evaluate.save(MODEL_EVAL_PATH, **results, **hyperparams)
        return results


