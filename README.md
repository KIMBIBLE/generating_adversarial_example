# generating_adversarial_example

## 1. How To Install

I recommend you to use python3

    git clone https://github.com/Oss9935/generating_adversarial_example.git

    cd ./generating_adversarial_example

    virtualenv —system-site-packages .

    source bin/activate

    python —version #(virtualenv에 파이썬 3 설정되어있나 확인)

    pip install -r requirements.txt


## 2. How To Use (example)

    1. 이미 트레이닝 된 모델로 특정 이미지에 대한 판별 결과만들기.
    python predict_image.py data/tiger.png

    2. 해당 모델과 이미지를 기반으로 adversarial example생성

    # default image object : starfish (327)
    python generate_adversarial_image.py ./data/tiger.png
    
    (or) 
    # You can set other ImageNet class numbers(in the ./imagenet_classes_map.txt)
    python generate_adversarial_image.py ./data/tiger.png 328
