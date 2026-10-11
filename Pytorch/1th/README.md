# 활성화 함수, XOR, 회귀: PyTorch 실험

`activation_xor_regression.py`는 활성화 함수의 출력과 기울기, XOR 분류,
`y=x²` 회귀, 학습된 은닉 뉴런의 기여, 학습 구간 밖 예측을 그래프로 확인합니다.

## VS Code에서 디버깅

1. VS Code에서 이 `1th` 폴더 자체를 엽니다.
2. Microsoft Python 확장과 Python Debugger 확장을 설치합니다.
3. `activation_xor_regression.py`의 원하는 줄 번호 왼쪽을 클릭해 중단점을 둡니다.
4. **실행 및 디버그**에서 **PyTorch: XOR + regression**을 선택하고 `F5`를 누릅니다.

`.vscode/launch.json`이 이 파일을 실행하며, `.venv/Scripts/python.exe`를 사용합니다.
처음부터 끝까지 실행하면 이 폴더의 `outputs/`에 그림 12장과
`experiment_metrics.json`이 생성됩니다. 경로를 바꾸려면 실행 인수로
`--output-dir 원하는_폴더`를 사용하세요.

### 중단점 추천

- `OneHiddenLayer.forward`: 아핀 변환과 활성화 함수의 순전파 확인
- `fit`의 `loss.backward()`: 역전파 전후의 기울기 확인
- `train_xor`: 선형 모델과 비선형 모델의 차이 확인
- `train_regression`: 회귀 모델 학습 확인
- `plot_regression`의 `model.parts(dense)`: 각 뉴런의 `z`, `h`, `contribution` 확인

`fit`은 학습 단계마다 호출되므로 모든 단계에서 멈추면 느립니다. 조건부 중단점을
사용하거나 처음 몇 단계만 확인한 뒤 중단점을 해제하면 편리합니다.

## 터미널 실행

이 폴더에서 PowerShell을 열고:

```powershell
.\.venv\Scripts\python.exe .\activation_xor_regression.py
```

가상 환경이 없거나 다시 만들려면:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 재현성 및 범위

- CPU 실행, 고정 난수 시드, 결정적인 작은 데이터셋을 사용합니다.
- XOR 학습 데이터는 네 점뿐이며, 점 사이의 경계는 모델에 따라 달라질 수 있습니다.
- 회귀 수치는 학습에 쓴 `[-2, 2]` 구간에서 계산했습니다.
- 버전에 따라 마지막 수치가 조금 달라질 수 있습니다.

## 은닉층과 출력층 비교 그래프

- `outputs/11_xor_hidden_to_output.png`: XOR의 ReLU·Tanh 은닉 뉴런 두 개와 최종 분류 확률을 같은 입력 평면에서 비교합니다. 출력층은 표시된 두 개를 포함한 은닉 뉴런 8개 모두를 사용합니다.
- `outputs/12_regression_hidden_to_output.png`: 은닉층 출력 `h_j(x)`, 가중 기여 `v_j h_j(x)`, 전체 회귀 예측을 동일한 입력 축에서 비교합니다. 출력층은 은닉 뉴런 16개 모두를 사용합니다.
