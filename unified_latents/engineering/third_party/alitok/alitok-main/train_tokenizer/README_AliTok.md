## Implementation of Tokenizer Training

### ⛺ Important Notes for Training
During the tokenizer training process, we manually adjusted the learning rate and decided when to initiate GAN training based on the decrease in training loss. Therefore, you might need to manually adjust these parameters by restarting the training task. Upon each restart, the code will automatically resume from ```last.pth``` if this file is exists. We provide the number of training steps for each phase as a reference, you can decide whether to end the current training phase early when the loss converges.


### ✏️ Usage <a name="4"></a> 
1. Prepare

    Download ImageNet Dataset and [imagenet_train.txt](https://drive.google.com/file/d/1uDrhK2nvLgUKUYAC4YZsOMdjNs5reKeK/view?usp=sharing), put imagenet_train.txt in the `train_tokenizer` folder. 
    ```
    cd train_tokenizer
    ```

2. Stage 1 of Tokenizer Training (Bi Enc + Causal Dec)

    ① Although we ultimately adopted Perceptual loss rather than LPIPS loss, including LPIPS loss in the early training steps can effectively prevent reconstruction results from being blank or noisy. This training phase typically continues for 50K steps until the reconstructed images appear normal. The reconstruction images will be generated every 500 steps in ```logs/vae_2025-XX-XX_XX-XX-XX/samples``` 
    
    ```
    python -m torch.distributed.launch train_tokenizer.py --root 'datasets/imagenet/train' --perceptual_weight 0.1 --lpips_weight 1.0 --clustering_vq
    ```

    ② We then removed the LPIPS loss and gradually reduced the learning rate from 1e-4 to 2e-5. You will need to manually adjust the learning rate based on the decrease in training loss. This phase of training generally spans approximately 200K steps.

    ```
    python -m torch.distributed.launch train_tokenizer.py --root 'datasets/imagenet/train' --lr XXX --clustering_vq
    ```

    ③ Add GAN loss. To immediately begin GAN training, you can set disc_start to 0. The learning rate should then be gradually lowered from 2e-5 to 2e-6 throughout the training process. This phase of training generally spans approximately 100K steps.

    ```
    python -m torch.distributed.launch train_tokenizer.py --root 'datasets/imagenet/train' --disc_start 0 --lr XXX --clustering_vq
    ```

2. Stage 2 of Tokenizer Training (Bi Enc + Bi Dec), similar to Stage 1

    ① Around Around 50K steps.
    
    ```
    python -m torch.distributed.launch train_tokenizer.py --root 'datasets/imagenet/train' --stage stage2 --perceptual_weight 0.1 --lpips_weight 1.0 
    ```

    ② Reduce the learning rate from 1e-4 to 2e-5. Around 150K steps.
    ```
    python -m torch.distributed.launch train_tokenizer.py --root 'datasets/imagenet/train' --stage stage2 --lr XXX
    ```

    ③ Reduce the learning rate from 2e-5 to 2e-6 and add GAN loss. Around 100K steps.
    ```
    python -m torch.distributed.launch train_tokenizer.py --root 'datasets/imagenet/train' --stage stage2 --lr XXX
    ```
