```python
!pip install tensorboardX
```

    Collecting tensorboardX
      Downloading tensorboardX-2.1-py2.py3-none-any.whl (308 kB)
    Collecting protobuf>=3.8.0
      Downloading protobuf-3.14.0-py2.py3-none-any.whl (173 kB)
    Requirement already satisfied: six in c:\anaconda3\lib\site-packages (from tensorboardX) (1.15.0)
    Requirement already satisfied: numpy in c:\anaconda3\lib\site-packages (from tensorboardX) (1.19.2)
    Installing collected packages: protobuf, tensorboardX
    Successfully installed protobuf-3.14.0 tensorboardX-2.1
    


```python
!pip install jupyter-tensorboard
```

    Collecting jupyter-tensorboard
      Downloading jupyter_tensorboard-0.2.0.tar.gz (15 kB)
    Requirement already satisfied: notebook>=5.0 in c:\anaconda3\lib\site-packages (from jupyter-tensorboard) (6.1.4)
    Requirement already satisfied: prometheus-client in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (0.8.0)
    Requirement already satisfied: pyzmq>=17 in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (19.0.2)
    Requirement already satisfied: jupyter-core>=4.6.1 in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (4.6.3)
    Requirement already satisfied: jinja2 in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (2.11.2)
    Requirement already satisfied: ipython-genutils in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (0.2.0)
    Requirement already satisfied: argon2-cffi in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (20.1.0)
    Requirement already satisfied: jupyter-client>=5.3.4 in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (6.1.7)
    Requirement already satisfied: traitlets>=4.2.1 in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (5.0.5)
    Requirement already satisfied: nbconvert in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (6.0.7)
    Requirement already satisfied: ipykernel in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (5.3.4)
    Requirement already satisfied: Send2Trash in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (1.5.0)
    Requirement already satisfied: tornado>=5.0 in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (6.0.4)
    Requirement already satisfied: nbformat in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (5.0.8)
    Requirement already satisfied: terminado>=0.8.3 in c:\anaconda3\lib\site-packages (from notebook>=5.0->jupyter-tensorboard) (0.9.1)
    Requirement already satisfied: pywin32>=1.0; sys_platform == "win32" in c:\anaconda3\lib\site-packages (from jupyter-core>=4.6.1->notebook>=5.0->jupyter-tensorboard) (227)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\anaconda3\lib\site-packages (from jinja2->notebook>=5.0->jupyter-tensorboard) (1.1.1)
    Requirement already satisfied: cffi>=1.0.0 in c:\anaconda3\lib\site-packages (from argon2-cffi->notebook>=5.0->jupyter-tensorboard) (1.14.3)
    Requirement already satisfied: six in c:\anaconda3\lib\site-packages (from argon2-cffi->notebook>=5.0->jupyter-tensorboard) (1.15.0)
    Requirement already satisfied: python-dateutil>=2.1 in c:\anaconda3\lib\site-packages (from jupyter-client>=5.3.4->notebook>=5.0->jupyter-tensorboard) (2.8.1)
    Requirement already satisfied: testpath in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (0.4.4)
    Requirement already satisfied: jupyterlab-pygments in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (0.1.2)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (1.4.3)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (0.5.1)
    Requirement already satisfied: pygments>=2.4.1 in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (2.7.2)
    Requirement already satisfied: entrypoints>=0.2.2 in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (0.3)
    Requirement already satisfied: bleach in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (3.2.1)
    Requirement already satisfied: defusedxml in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (0.6.0)
    Requirement already satisfied: mistune<2,>=0.8.1 in c:\anaconda3\lib\site-packages (from nbconvert->notebook>=5.0->jupyter-tensorboard) (0.8.4)
    Requirement already satisfied: ipython>=5.0.0 in c:\anaconda3\lib\site-packages (from ipykernel->notebook>=5.0->jupyter-tensorboard) (7.19.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in c:\anaconda3\lib\site-packages (from nbformat->notebook>=5.0->jupyter-tensorboard) (3.2.0)
    Requirement already satisfied: pywinpty>=0.5 in c:\anaconda3\lib\site-packages (from terminado>=0.8.3->notebook>=5.0->jupyter-tensorboard) (0.5.7)
    Requirement already satisfied: pycparser in c:\anaconda3\lib\site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=5.0->jupyter-tensorboard) (2.20)
    Requirement already satisfied: nest-asyncio in c:\anaconda3\lib\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=5.0->jupyter-tensorboard) (1.4.2)
    Requirement already satisfied: async-generator in c:\anaconda3\lib\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=5.0->jupyter-tensorboard) (1.10)
    Requirement already satisfied: packaging in c:\anaconda3\lib\site-packages (from bleach->nbconvert->notebook>=5.0->jupyter-tensorboard) (20.4)
    Requirement already satisfied: webencodings in c:\anaconda3\lib\site-packages (from bleach->nbconvert->notebook>=5.0->jupyter-tensorboard) (0.5.1)
    Requirement already satisfied: decorator in c:\anaconda3\lib\site-packages (from ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (4.4.2)
    Requirement already satisfied: pickleshare in c:\anaconda3\lib\site-packages (from ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (0.7.5)
    Requirement already satisfied: backcall in c:\anaconda3\lib\site-packages (from ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (0.2.0)
    Requirement already satisfied: colorama; sys_platform == "win32" in c:\anaconda3\lib\site-packages (from ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (0.4.4)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in c:\anaconda3\lib\site-packages (from ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (3.0.8)
    Requirement already satisfied: setuptools>=18.5 in c:\anaconda3\lib\site-packages (from ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (50.3.1.post20201107)
    Requirement already satisfied: jedi>=0.10 in c:\anaconda3\lib\site-packages (from ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (0.17.1)
    Requirement already satisfied: attrs>=17.4.0 in c:\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat->notebook>=5.0->jupyter-tensorboard) (20.3.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in c:\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat->notebook>=5.0->jupyter-tensorboard) (0.17.3)
    Requirement already satisfied: pyparsing>=2.0.2 in c:\anaconda3\lib\site-packages (from packaging->bleach->nbconvert->notebook>=5.0->jupyter-tensorboard) (2.4.7)
    Requirement already satisfied: wcwidth in c:\anaconda3\lib\site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (0.2.5)
    Requirement already satisfied: parso<0.8.0,>=0.7.0 in c:\anaconda3\lib\site-packages (from jedi>=0.10->ipython>=5.0.0->ipykernel->notebook>=5.0->jupyter-tensorboard) (0.7.0)
    Building wheels for collected packages: jupyter-tensorboard
      Building wheel for jupyter-tensorboard (setup.py): started
      Building wheel for jupyter-tensorboard (setup.py): finished with status 'done'
      Created wheel for jupyter-tensorboard: filename=jupyter_tensorboard-0.2.0-py2.py3-none-any.whl size=15267 sha256=7e5841334060430fcfaed47ffbe6a02521d0c2e9963f971bcb21d342bff83038
      Stored in directory: c:\users\playdata\appdata\local\pip\cache\wheels\04\d8\14\cd409c6b7d6fd055ec050d65446498357abcec8d81bab11c21
    Successfully built jupyter-tensorboard
    Installing collected packages: jupyter-tensorboard
    Successfully installed jupyter-tensorboard-0.2.0
    


```python
!pip install tensorflow
```

    Collecting tensorflow
      Downloading tensorflow-2.4.1-cp38-cp38-win_amd64.whl (370.7 MB)
    Requirement already satisfied: h5py~=2.10.0 in c:\anaconda3\lib\site-packages (from tensorflow) (2.10.0)
    Collecting gast==0.3.3
      Downloading gast-0.3.3-py2.py3-none-any.whl (9.7 kB)
    Collecting opt-einsum~=3.3.0
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    Collecting flatbuffers~=1.12.0
      Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: numpy~=1.19.2 in c:\anaconda3\lib\site-packages (from tensorflow) (1.19.2)
    Collecting wrapt~=1.12.1
      Using cached wrapt-1.12.1.tar.gz (27 kB)
    Collecting termcolor~=1.1.0
      Downloading termcolor-1.1.0.tar.gz (3.9 kB)
    Collecting absl-py~=0.10
      Downloading absl_py-0.11.0-py3-none-any.whl (127 kB)
    Requirement already satisfied: six~=1.15.0 in c:\anaconda3\lib\site-packages (from tensorflow) (1.15.0)
    Requirement already satisfied: wheel~=0.35 in c:\anaconda3\lib\site-packages (from tensorflow) (0.35.1)
    Requirement already satisfied: protobuf>=3.9.2 in c:\anaconda3\lib\site-packages (from tensorflow) (3.14.0)
    Collecting keras-preprocessing~=1.1.2
      Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    Collecting grpcio~=1.32.0
      Downloading grpcio-1.32.0-cp38-cp38-win_amd64.whl (2.6 MB)
    Collecting tensorflow-estimator<2.5.0,>=2.4.0
      Downloading tensorflow_estimator-2.4.0-py2.py3-none-any.whl (462 kB)
    Collecting tensorboard~=2.4
      Downloading tensorboard-2.4.1-py3-none-any.whl (10.6 MB)
    Collecting google-pasta~=0.2
      Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
    Collecting astunparse~=1.6.3
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: typing-extensions~=3.7.4 in c:\anaconda3\lib\site-packages (from tensorflow) (3.7.4.3)
    Requirement already satisfied: werkzeug>=0.11.15 in c:\anaconda3\lib\site-packages (from tensorboard~=2.4->tensorflow) (1.0.1)
    Collecting google-auth<2,>=1.6.3
      Downloading google_auth-1.26.1-py2.py3-none-any.whl (116 kB)
    Requirement already satisfied: setuptools>=41.0.0 in c:\anaconda3\lib\site-packages (from tensorboard~=2.4->tensorflow) (50.3.1.post20201107)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\anaconda3\lib\site-packages (from tensorboard~=2.4->tensorflow) (2.24.0)
    Collecting tensorboard-plugin-wit>=1.6.0
      Downloading tensorboard_plugin_wit-1.8.0-py3-none-any.whl (781 kB)
    Collecting markdown>=2.6.8
      Downloading Markdown-3.3.3-py3-none-any.whl (96 kB)
    Collecting google-auth-oauthlib<0.5,>=0.4.1
      Downloading google_auth_oauthlib-0.4.2-py2.py3-none-any.whl (18 kB)
    Collecting rsa<5,>=3.1.4; python_version >= "3.6"
      Downloading rsa-4.7-py3-none-any.whl (34 kB)
    Collecting cachetools<5.0,>=2.0.0
      Downloading cachetools-4.2.1-py3-none-any.whl (12 kB)
    Collecting pyasn1-modules>=0.2.1
      Downloading pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in c:\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in c:\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow) (1.25.11)
    Collecting requests-oauthlib>=0.7.0
      Downloading requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)
    Collecting pyasn1>=0.1.3
      Downloading pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
    Collecting oauthlib>=3.0.0
      Downloading oauthlib-3.1.0-py2.py3-none-any.whl (147 kB)
    Building wheels for collected packages: wrapt, termcolor
      Building wheel for wrapt (setup.py): started
      Building wheel for wrapt (setup.py): finished with status 'done'
      Created wheel for wrapt: filename=wrapt-1.12.1-py3-none-any.whl size=19558 sha256=80af1b79db1f1eeea0aeea99b9c2bb2c950a70d2afa0ff34973a20d226d0782b
      Stored in directory: c:\users\playdata\appdata\local\pip\cache\wheels\5f\fd\9e\b6cf5890494cb8ef0b5eaff72e5d55a70fb56316007d6dfe73
      Building wheel for termcolor (setup.py): started
      Building wheel for termcolor (setup.py): finished with status 'done'
      Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4835 sha256=49b5ec3d0c964c40274ea59514cce62bb7db006f3dd66128ae70da72a4223b2c
      Stored in directory: c:\users\playdata\appdata\local\pip\cache\wheels\a0\16\9c\5473df82468f958445479c59e784896fa24f4a5fc024b0f501
    Successfully built wrapt termcolor
    Installing collected packages: gast, opt-einsum, flatbuffers, wrapt, termcolor, absl-py, keras-preprocessing, grpcio, tensorflow-estimator, pyasn1, rsa, cachetools, pyasn1-modules, google-auth, tensorboard-plugin-wit, markdown, oauthlib, requests-oauthlib, google-auth-oauthlib, tensorboard, google-pasta, astunparse, tensorflow
      Attempting uninstall: wrapt
        Found existing installation: wrapt 1.11.2
        Uninstalling wrapt-1.11.2:
          Successfully uninstalled wrapt-1.11.2
    Successfully installed absl-py-0.11.0 astunparse-1.6.3 cachetools-4.2.1 flatbuffers-1.12 gast-0.3.3 google-auth-1.26.1 google-auth-oauthlib-0.4.2 google-pasta-0.2.0 grpcio-1.32.0 keras-preprocessing-1.1.2 markdown-3.3.3 oauthlib-3.1.0 opt-einsum-3.3.0 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-oauthlib-1.3.0 rsa-4.7 tensorboard-2.4.1 tensorboard-plugin-wit-1.8.0 tensorflow-2.4.1 tensorflow-estimator-2.4.0 termcolor-1.1.0 wrapt-1.12.1
    


```python
!pip install keras
```

    Collecting keras
      Downloading Keras-2.4.3-py2.py3-none-any.whl (36 kB)
    Requirement already satisfied: h5py in c:\anaconda3\lib\site-packages (from keras) (2.10.0)
    Requirement already satisfied: numpy>=1.9.1 in c:\anaconda3\lib\site-packages (from keras) (1.19.2)
    Requirement already satisfied: scipy>=0.14 in c:\anaconda3\lib\site-packages (from keras) (1.5.2)
    Requirement already satisfied: pyyaml in c:\anaconda3\lib\site-packages (from keras) (5.3.1)
    Requirement already satisfied: six in c:\anaconda3\lib\site-packages (from h5py->keras) (1.15.0)
    Installing collected packages: keras
    Successfully installed keras-2.4.3
    


```python
!pip install h5py
```

    Requirement already satisfied: h5py in c:\anaconda3\lib\site-packages (2.10.0)
    Requirement already satisfied: numpy>=1.7 in c:\anaconda3\lib\site-packages (from h5py) (1.19.2)
    Requirement already satisfied: six in c:\anaconda3\lib\site-packages (from h5py) (1.15.0)
    


```python
!python3 -m pip install --upgrade tensorflow
```

    Python
    


```python
# 1. 임포트
import tensorflow as tf
tf.__version__

```




    '2.4.1'




```python
# 2. 행과 열을 만들자
matrix1 = tf.constant([[3,3] ])
matrix1

matrix2 = tf.constant([[2],[2]])

res = tf.matmul(matrix1, matrix2)
print(res)

#tf.constant([1,2,3,4,5,6])

x = tf.constant(([1,2,3,4]))
res02 = tf.math.multiply(x,x)

hap = tf.math.add(x,x)
print(hap)


```

    tf.Tensor([[12]], shape=(1, 1), dtype=int32)
    tf.Tensor([2 4 6 8], shape=(4,), dtype=int32)
    


```python
# 행렬 연산을 구현해보자 3.0 2.0 5.0 3.0*(2.0+5.0)
input01 = tf.constant([3.0])
input02 = tf.constant([2.0])
input03 = tf.constant([5.0])

hap = tf.add(input02, input03)
res = tf.multiply(input01, hap)
print(hap)
print(res)


res02 = tf.multiply(input01, tf.add(input02, input03))
print(res02)
```

    tf.Tensor([7.], shape=(1,), dtype=float32)
    tf.Tensor([21.], shape=(1,), dtype=float32)
    tf.Tensor([21.], shape=(1,), dtype=float32)
    


```python
#3. 즉시 기본실행
tf.executing_eagerly()
x =[[2]]
m = tf.matmul(x,x)
print("hello.[]",format(m))


```

    hello.[] [[4]]
    


```python
#4. 반환 작업
a = tf.constant([[1,2], [3,4]])
print(a)
print(a.shape)
print(a.dtype, type(a.dtype))
print(a.numpy(), type(a.numpy()))
print(type(a))
```

    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32)
    (2, 2)
    <dtype: 'int32'> <class 'tensorflow.python.framework.dtypes.DType'>
    [[1 2]
     [3 4]] <class 'numpy.ndarray'>
    <class 'tensorflow.python.framework.ops.EagerTensor'>
    


```python
# 5. 브로드캐스팅 (broadcasting)
b = tf.add(a, 1)
print(b)

#6. 연산자 오버로딩 지원
print(a*b)


```

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32)
    tf.Tensor(
    [[ 2  6]
     [12 20]], shape=(2, 2), dtype=int32)
    


```python
# 7. Numoy 지원
import numpy as np
res = np.multiply(a,b)
print(res, type(res))

```

    [[ 2  6]
     [12 20]] <class 'numpy.ndarray'>
    


```python
# 8. Variable
name = tf.Variable("홍길동", tf.string)
kor = tf.Variable(100, tf.int32)
eng = tf.Variable(90, tf.int32)
tot = kor + eng
print(name.numpy())
print(tot.numpy())


```

    b'\xed\x99\x8d\xea\xb8\xb8\xeb\x8f\x99'
    190
    


```python
# 9. tensor 차웜을 리턴받자. tf.zeros(), tf.rank()
# 배치*높이*너비*색상
my_img = tf.zeros([10,299,299,3])
res = tf.rank(my_img)
print(res)
res
```

    tf.Tensor(4, shape=(), dtype=int32)
    




    <tf.Tensor: shape=(), dtype=int32, numpy=4>




```python
#10. 요소를 리턴
my_v = tf.Variable([1,2,3,4])
my_s = my_v[2]
print(my_s.numpy() )

```

    3
    


```python
#11. tf의 객체를 Variable로 연동해보자
my_val = tf.Variable(tf.zeros([2,2,3]))
my_val.numpy()
```




    array([[[0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.]]], dtype=float32)




```python
#12. 변수 자동 변환
v = tf.Variable(0.0)
v.numpy()

w = v+1 # w는 v값을 기준으로 계산되는 tf.Tensor, 변수가 수식으로 사용하게 되면 tf.Tensor로 변환
print(w)

v = tf.Variable(0.0)
v.assign_add(1)
print(v.read_value())
print(v)
```

    tf.Tensor(1.0, shape=(), dtype=float32)
    tf.Tensor(1.0, shape=(), dtype=float32)
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
    


```python
"텐서"="스칼라" 또는 "벡터" 또는 "매트릭스"
```


```python
import tensorflow as tf
#3*2행의 float

A = tf.Variable([[1,2],[3,4],[5,6]], dtype = tf.float32)
print(A.numpy())

# 전체를 -값으로 변경
A.assign([[-1,-2],[-3,-4],[-5,-6]])
print(A.numpy())

# 3*2 행의 값을 모두 0으로 초기화 float32
A = tf.Variable(tf.zeros([3,2]), dtype = tf.float32)
print(A.numpy())

#3*2의 행의 값을 요소 모두를 평균은 0, 표준편차 0.1의 정규난수로 초기화를 시켜보자
# tf.random.normal()
A = tf.Variable(tf.random.normal([3,2], mean =0.0, stddev = 0.1, dtype = tf.dtypes.float32))
print(A.numpy())

# 3*2의 행의 값 요수모두 -1~3까지의 범위에서 선택
A = tf.Variable(tf.random.uniform([3,2], minval =-1, maxval = 3, dtype = tf.dtypes.float32))
print(A.numpy())




```

    [[1. 2.]
     [3. 4.]
     [5. 6.]]
    [[-1. -2.]
     [-3. -4.]
     [-5. -6.]]
    [[0. 0.]
     [0. 0.]
     [0. 0.]]
    [[-0.04262998 -0.20142844]
     [ 0.03829199 -0.09723001]
     [ 0.03557634  0.17470682]]
    [[ 2.6858053   1.3038964 ]
     [-0.7460766   2.1513114 ]
     [ 0.6756749  -0.58954334]]
    


```python
# 13. f(x) 표현하는 연산 시스템
# log(x), exp(x), sin(x), cos(x), sigmoid(), tanh(x) 등

A = tf.constant([1,2,3], tf.float32)
B = tf.math.log(A)
print(B.numpy())

B = tf.math.sigmod(A)
print(B.numpy())

B = tf.math.tanh(A)
print(B.numpy())

```

    [0.        0.6931472 1.0986123]
    [0.7310586  0.8807971  0.95257413]
    [0.7615942 0.9640276 0.9950547]
    


```python
# 14. softmax : 로지스틱 함수를 다차원으로 만든 것
# 다항 로지스틱 회귀로 출력에 대한 확률분포로 네트워크 출력을 하는 정규화
# 신경망의 활성화 함수로 자주 사용한다 (one - hot)
# z = K로 실수값을 벡터를 받아서 입력숫자에 지수에 비례하는 K개의 확률분포 정규화 시킨것

tf.nn.softmax(logits, axis = None, name=None)
softmax = tf.exp(logits) . tf.reduce_sum(tf.exp(logits), axis)

tf.nn.softmax(x)
# 입력인수 x는 텐서, 정수형 텐서(상수), 변수 상수
# 출력 값1. x의 각 요소마다 exp90를 계산해서 A에 할당
#        2. A의 요소를 모두 더해서 임의 변수 a에 넣는다.
#        3. A의 요소를 a로 나눠서 출력 텐서를 만든다.
#        4. 출력텐서의 각 요소의 값은 확률을 나타낸다.



```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-46-ff4797d61ee6> in <module>
          4 # z = K로 실수값을 벡터를 받아서 입력숫자에 지수에 비례하는 K개의 확률분포 정규화 시킨것
          5 
    ----> 6 tf.nn.softmax(logits, axis = None, name=None)
          7 softmax = tf.exp(logits) . tf.reduce_sum(tf.exp(logits), axis)
          8 
    

    NameError: name 'logits' is not defined



```python

A = tf.constant([1,2,3], tf.float32)
B = tf.nn.softmax(A)
print(B.numpy()) # 확률분포

B = tf.reduce_sum(A)
print(B.numpy())


```

    [0.09003057 0.24472848 0.66524094]
    6.0
    


```python

@tf.function

def myOP(a,b,c):
    t = tf.math.add(a,b)
    return tf.math.multiply(t,c)


A = tf.constant([1,2], tf.float32)
B = tf.constant([3,4], tf.float32)
C = tf.constant([5,6], tf.float32)

D = myOP(A, B, C)
print(D, type(D))
print(D.numpy())
```

    tf.Tensor([20. 36.], shape=(2,), dtype=float32) <class 'tensorflow.python.framework.ops.EagerTensor'>
    [20. 36.]
    


```python
#16. 사용자 자료형 만들기
# tf.Module를 상속받아 구현한다
# tf.Module 인스턴스는 Variable
# trainable_variables = 훈련가능한 변수를 리턴

class MyModuleOne(tf.Module):
    def __init__(self):
        self.VO = tf.Variable(1.0)
        self.VS = [tf.Variable(x) for x in range(10)]

class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10.0)

m = MyOtherModule()
len(m.variables)
print(m.variables)
print(m.m)# 11은 m.m에서 다른값은 m.v에서
print(m.v) 

```

    (<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=10.0>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=0>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=1>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=2>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=4>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=5>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=6>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=7>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=8>, <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=9>)
    <__main__.MyModuleOne object at 0x000002C6CBD1A190>
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=10.0>
    


```python
#17. 케라스
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
model


```




    <tensorflow.python.keras.engine.sequential.Sequential at 0x2c6cbd74430>




```python
# 18. 순차모델을 만들어 보자
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, name="Layer_0", input_shape = (1000,)),
    tf.keras.layers.Dense(10, name="Layer_1"),
    tf.keras.layers.Dense(1, name="Layer_2")
])

model.summary()
print(type(model.layers[0]))
print(model.layers[0].name)

d = {k.name: i for i,k in enumerate(model.layers)}
print(d)
print(d["Layer_1"])
print(d.get("Layer_1"))

layer_name = [k.name for k in model.layers]
print(layer_name)


```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Layer_0 (Dense)              (None, 100)               100100    
    _________________________________________________________________
    Layer_1 (Dense)              (None, 10)                1010      
    _________________________________________________________________
    Layer_2 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 101,121
    Trainable params: 101,121
    Non-trainable params: 0
    _________________________________________________________________
    <class 'tensorflow.python.keras.layers.core.Dense'>
    Layer_0
    {'Layer_0': 0, 'Layer_1': 1, 'Layer_2': 2}
    1
    1
    ['Layer_0', 'Layer_1', 'Layer_2']
    


```python
심층 신경망은 학습한 모든 변환을 수치 데이터 텐서에 적용하는 텐서연산을 사용한다
ex) 텐서 덧셈, 텐서 곱셈

tf.keras.layers.Dense(1, ㅜ믇="Layer_2")-> 신경망을 만들었다. 층을 설정했다

# 19. 간단한 공식을 이용해서 값을 전달한 후 평가로 예측값을 구현해보자


```


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
# y= ax+b
# 5 = a2+ b3 = a1+b7 = a*3 + b


# [1단계] 데이터
x = np.array([1,2,3,4,5])
y = x*2 + 1
print(x)
print(y)

# [2단계] 학습구조모델링
model= Sequential() # 레이어를 층층이 쌍하주는 메소드
model.add( Dense(1, input_shape = (1,)))

model.compile('SGD', 'mse')

#SGD (Stochastic gradient descent :확률적 경사하강법)
# mse (Mean Squrared Error : 평균제곱오차)

# [3단계] 학습수행
model.fit(x,y, epochs=100, verbose = 0)

# [4단계] 평가
print( 'y', y, 'predict : ', model.predict(x).flatten())


```

    [1 2 3 4 5]
    [ 3  5  7  9 11]
    y [ 3  5  7  9 11] predict :  [ 2.9499693  4.9691358  6.988302   9.007468  11.026634 ]
    


```python
#20. tf.GradientTape API : 텐서플로는 자동미분하는 API를 제공한다
#context 안에 실행된 모든 연산을 tape에 기록한다
# 후진방식 자동미분을 사용해서 tape에 기록된 연산을 결과에 계산한다
# 기울기를 구하는 클래스
# 예상한 값과 실제 결과를 비교해서 가능한 차이가 적게 매개변수를 조정하게 된다.

x = tf.ones((2,2))
x

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z= tf.multiply(y, y)

dz_dx = t.gradient(z, x) # 입력텐서 x에 대한 z 값
print(dz_dx)
    
for i in [0,1]:
    for j in [0,1]:
        assert dz_dx[i][j] == 8.0
        dz_dx[i][j].numpy() == 8.0
        print(dz_dx[i][j])
    
```

    tf.Tensor(
    [[8. 8.]
     [8. 8.]], shape=(2, 2), dtype=float32)
    tf.Tensor(8.0, shape=(), dtype=float32)
    tf.Tensor(8.0, shape=(), dtype=float32)
    tf.Tensor(8.0, shape=(), dtype=float32)
    tf.Tensor(8.0, shape=(), dtype=float32)
    


```python

```


```python

```


```python

```
