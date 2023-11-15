from transformers import (
    GPT2TokenizerFast,
    AdamW,
    get_scheduler
)
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn 
import pandas as pd
from model import GPT2PromptTuningLM
from data import CSVDataset
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from dataclasses import dataclass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    num_train_epochs = 10
    batch_size = 4
    weight_decay = 0.01
    learning_rate = 0.01
    lr_scheduler_type = "linear"
    num_warmup_steps = 0
    max_train_steps = 10
    is_train = True
    gradient_accumulation_steps = 4
    patience = 2
    subset_size = 25000
    
def train_loop(model, optimiser, num_epochs, train_loader, dev_loader, lr_scheduler, grad_acc_steps, patience):
    best_loss = float('inf')
    is_improved = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc='finetuning')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else None
            optimiser.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Compute the loss
            loss = outputs.loss
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            
            # # Gradient accumulation
            if (i + 1) % grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                lr_scheduler.step()

        # Print the average loss after each epoch
        average_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] finished with average loss: {average_loss:.4f}')
        
        # Validation
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                input_ids_val = batch['input_ids'].to(device)
                attention_mask_val = batch['attention_mask'].to(device)
                labels_val = batch['labels'].to(device) if 'labels' in batch else None

                outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val, labels=labels_val)
                loss_val = outputs_val.loss
                dev_loss += loss_val.item()

            average_dev_loss = dev_loss / len(dev_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}] finished with average validation loss: {average_dev_loss:.4f}')

            # Early stopping
            if average_dev_loss < best_loss:
                best_loss = average_dev_loss
                is_improved = 0
            else:
                is_improved += 1

            # if is_improved >= patience:
            #     print(f'No improvement for {patience} consecutive epochs. Early stopping.')
            #     break
    return model

def test_loop(model, dataloader, tokenizer):
    model.eval()
    references = []
    hypotheses = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing model")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)
        labels = labels.cpu().numpy()

        # Convert predictions and labels to text
        predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
        reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

        hypotheses.append(predicted_text)
        references.append(reference_text)

    # Calculate BLEU score
    bleu_score = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses])

    # Calculate METEOR score
    meteor_avg_score = meteor_score(references, hypotheses)

    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, references, avg=True)

    print(f'BLEU Score: {bleu_score * 100:.2f}')
    print(f'METEOR Score: {meteor_avg_score * 100:.2f}')
    print(f'ROUGE Scores: {rouge_scores}')
    
if __name__ == '__main__':
    args = Config()
    print({k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)})
    tokeniser = GPT2TokenizerFast.from_pretrained('gpt2')
    tokeniser.pad_token = tokeniser.eos_token
    print('declaring model')
    model = GPT2PromptTuningLM.from_pretrained(
        'gpt2',
        tokeniser,
        prompt_str='[QA]'
    ).to(device)
    
    print('tokenising train file')
    train_file = 'dataset/train.json'
    dataset = CSVDataset(train_file, tokeniser, max_length=768, subset_size=args.subset_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print('tokenising val file')
    validation_file = 'dataset/dev.json'
    dataset_val = CSVDataset(validation_file, tokeniser, max_length=768, subset_size=int(args.subset_size/10))
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    
    # Only update soft prompt'weights for prompt-tuning. ie, all weights in LM are set as `require_grad=False`. 
    optimiser_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": args.weight_decay,
        }
    ]
    optimiser = AdamW(optimiser_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimiser,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    if args.is_train:
        print('training model')
        model = train_loop(model, optimiser,args.num_train_epochs,dataloader, dataloader_val, lr_scheduler, args.gradient_accumulation_steps, args.patience)
        
        print('saving model')
        torch.save(model.state_dict(), f'../models/qa.pth')
    else:
        print('evaluating model')
        model.load_state_dict(torch.load(f'../models/qa.pth'))
        
        print('evaluating train set')
        test_loop(model, dataloader, tokeniser)
        
        print('evaluating test set')
        test_loop(model, dataloader_val, tokeniser)
        
    print('khattam shud')
