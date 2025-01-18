# Text Generation with GPT

This project implements a text generation application using a GPT model. The application allows users to input a prompt and generate multiple variations of text based on that prompt.
This is available on HuggingFace Spaces also : https://huggingface.co/spaces/peeyushsinghal/GPT2_Demo 
![App Screenshot](Gradio_SS.png)

## Features

- Enter a prompt in a text box.
- Select the number of sequences to generate (1 to 5).
- View each generated sequence clearly labeled.


## Usage

1. Clone the repository:

```bash
git clone https://github.com/peeyushsinghal/shakespeare-style-text-generation.git
cd shakespeare-style-text-generation
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Gradio application:

```bash
python src/app.py
```

4. Open your web browser and navigate to the URL provided in the terminal (usually `http://127.0.0.1:7860`).


5. Usage:

- Enter your desired prompt in the input box.
- Use the slider to select the number of text sequences you want to generate.
- Click the "Submit" button to see the generated text sequences.

## Features

### Text Generation
- Context-aware text completion
- Multiple sequence generation
- Controllable generation length
- Preserves input prompt in output

### Technical Features
- Half-precision storage
- GPU/CPU compatibility
- Streamlit interface
- Token count display

## Model Training Logs 
```
Using device: mps

Model Architecture:
==================================================
Block Size (Context Length): 1024
Vocabulary Size: 50257
Number of Layers: 12
Number of Heads: 12
Embedding Dimension: 768
Dropout: 0.1

Parameter Counts:
--------------------------------------------------
Token Embeddings: 38,597,376
Position Embeddings: 786,432
Per Layer: 7,077,888
All Layers: 84,934,656
Total Trainable Parameters: 124,439,808

Estimated Model Size:
Full Precision (MB): 474.70
Half Precision (MB): 237.35
==================================================

loaded 338025 tokens
1 epoch = 82 batches
step0, loss: 10.9682, best loss: inf, moving avg loss: 10.9682, lr: 1.00e-04
Model saved at step 100
step100, loss: 5.6070, best loss: 5.6070, moving avg loss: 6.6620, lr: 9.99e-05
Model saved at step 101
Model saved at step 103
Model saved at step 113
Model saved at step 151
Model saved at step 152
Model saved at step 159
Model saved at step 161
Model saved at step 169
Model saved at step 180
Model saved at step 185
step200, loss: 4.9974, best loss: 4.6786, moving avg loss: 5.3583, lr: 9.96e-05
Model saved at step 233
Model saved at step 234
Model saved at step 241
Model saved at step 242
Model saved at step 243
Model saved at step 262
Model saved at step 267
step300, loss: 4.8728, best loss: 4.3879, moving avg loss: 4.8982, lr: 9.92e-05
Model saved at step 316
Model saved at step 323
Model saved at step 324
Model saved at step 349
Model saved at step 398
step400, loss: 4.5101, best loss: 4.1528, moving avg loss: 4.6134, lr: 9.86e-05
Model saved at step 405
Model saved at step 406
Model saved at step 431
Model saved at step 487
Model saved at step 488
step500, loss: 4.1066, best loss: 3.7987, moving avg loss: 4.3353, lr: 9.78e-05
Model saved at step 513
Model saved at step 569
Model saved at step 570
Model saved at step 590
Model saved at step 593
Model saved at step 595
step600, loss: 3.9333, best loss: 3.5487, moving avg loss: 4.0746, lr: 9.68e-05
Model saved at step 652
Model saved at step 672
Model saved at step 677
step700, loss: 3.8424, best loss: 3.3086, moving avg loss: 3.8114, lr: 9.57e-05
Model saved at step 735
Model saved at step 754
Model saved at step 757
Model saved at step 759
step800, loss: 3.5986, best loss: 3.0877, moving avg loss: 3.5605, lr: 9.44e-05
Model saved at step 817
Model saved at step 836
Model saved at step 837
Model saved at step 839
Model saved at step 841
Model saved at step 899
step900, loss: 3.5253, best loss: 2.8213, moving avg loss: 3.2594, lr: 9.30e-05
Model saved at step 918
Model saved at step 923
Model saved at step 981
step1000, loss: 2.6096, best loss: 2.5691, moving avg loss: 2.9883, lr: 9.14e-05
Model saved at step 1005
Model saved at step 1062
Model saved at step 1063
Model saved at step 1087
step1100, loss: 2.5744, best loss: 2.2297, moving avg loss: 2.7100, lr: 8.97e-05
Model saved at step 1145
Model saved at step 1169
step1200, loss: 2.5762, best loss: 2.0349, moving avg loss: 2.4592, lr: 8.78e-05
Model saved at step 1227
Model saved at step 1251
step1300, loss: 2.0482, best loss: 1.8736, moving avg loss: 2.2568, lr: 8.58e-05
Model saved at step 1309
Model saved at step 1333
Model saved at step 1391
step1400, loss: 1.7758, best loss: 1.6499, moving avg loss: 2.0197, lr: 8.37e-05
Model saved at step 1415
Model saved at step 1473
Model saved at step 1493
Model saved at step 1497
step1500, loss: 1.7530, best loss: 1.3564, moving avg loss: 1.7769, lr: 8.14e-05
Model saved at step 1554
Model saved at step 1555
Model saved at step 1575
Model saved at step 1579
step1600, loss: 1.5138, best loss: 1.1743, moving avg loss: 1.5549, lr: 7.91e-05
Model saved at step 1637
Model saved at step 1657
Model saved at step 1661
Model saved at step 1671
step1700, loss: 1.4449, best loss: 1.0219, moving avg loss: 1.3650, lr: 7.67e-05
Model saved at step 1719
Model saved at step 1738
Model saved at step 1739
Model saved at step 1743
Model saved at step 1753
step1800, loss: 0.8981, best loss: 0.8655, moving avg loss: 1.1498, lr: 7.41e-05
Model saved at step 1801
Model saved at step 1821
Model saved at step 1825
Model saved at step 1835
Model saved at step 1883
step1900, loss: 0.9042, best loss: 0.6744, moving avg loss: 0.9591, lr: 7.15e-05
Model saved at step 1903
Model saved at step 1907
Model saved at step 1917
Model saved at step 1965
Model saved at step 1985
Model saved at step 1989
Model saved at step 1999
step2000, loss: 0.6119, best loss: 0.4619, moving avg loss: 0.7612, lr: 6.89e-05
Model saved at step 2047
Model saved at step 2067
Model saved at step 2071
Model saved at step 2080
Model saved at step 2081
step2100, loss: 0.4643, best loss: 0.3520, moving avg loss: 0.6088, lr: 6.62e-05
Model saved at step 2149
Model saved at step 2153
Model saved at step 2162
Model saved at step 2163
step2200, loss: 0.4443, best loss: 0.2691, moving avg loss: 0.4910, lr: 6.34e-05
Model saved at step 2231
Model saved at step 2244
Model saved at step 2245
step2300, loss: 0.4089, best loss: 0.2134, moving avg loss: 0.3686, lr: 6.06e-05
Model saved at step 2313
Model saved at step 2317
Model saved at step 2325
Model saved at step 2326
Model saved at step 2395
step2400, loss: 0.2283, best loss: 0.1571, moving avg loss: 0.2671, lr: 5.78e-05
Model saved at step 2407
Model saved at step 2408
Model saved at step 2409
Model saved at step 2477
Model saved at step 2489
Model saved at step 2490
Model saved at step 2491
step2500, loss: 0.1304, best loss: 0.1033, moving avg loss: 0.1974, lr: 5.50e-05
Model saved at step 2559
Model saved at step 2562
Model saved at step 2563
Model saved at step 2571
Model saved at step 2572
step2600, loss: 0.1451, best loss: 0.0819, moving avg loss: 0.1558, lr: 5.21e-05
Model saved at step 2619
Model saved at step 2641
Model saved at step 2653
Model saved at step 2654
step2700, loss: 0.0986, best loss: 0.0615, moving avg loss: 0.1236, lr: 4.93e-05
Model saved at step 2723
---Target loss reached at step 2738
Training completed. Best loss: 0.0542, final loss: 0.0688
Model saved to checkpoint/model_final.pth
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (h): ModuleList(
      (0-11): 12 x Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=768, out_features=2304, bias=True)
          (c_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): MLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): GELU(approximate='tanh')
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```
