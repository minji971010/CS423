# CS423

# Setting
Python 3.7
```
pip install -r requirements.txt
```

# main2.py usage

1. 새로운 데이터 셋이 추가되면 sample_cfg 리스트에 추가를 해주세요.

예를 들어,
* 샘플 이미지들 (.jpg or .jpeg)이 samples/iron_sample/1.jpg 와 같이 있다.
  * -> "samples_dir": [ "samples/iron_sample/" ]
* 유사도 셋의 정보를 담은 npy 파일이 npy/iron.npy 이다.
  * -> "npy": "npy/iron.npy"
* 유사도 값이 n차원 벡터이다.
  * -> "dim": n (이러면 n개의 값들 각각을 기준으로 해서 page rank 알고리즘을 돌립니다.)
* 사용한 이미지의 수가 n개이다.
  * -> "img_cnt": n
  
```python
sample_cfg = [
    {
      "sample_dirs": [
          "./samples/iron_sample/"
      ],
      "npy": "npy/iron.npy",
      "dim": 3,
      "img_cnt": 50
    },
]
```

2. sample_cfg 리스트에 데이터 셋의 정보를 추가했으면, sample_cfg 아래에 있는 sample_idx 를 고쳐서 사용할 데이터 셋을 선택해주세요.
3. 맨 아래에 이런 부분이 있습니다.
```python
temp_list = list(reversed(sorted(rank_list)))[:10]
for val in temp_list:
    print(imgs[rank_list.index(val)])
```
이 부분은 그냥 이미지 10개를 rank 높았던 순서대로 출력하는 것입니다.
