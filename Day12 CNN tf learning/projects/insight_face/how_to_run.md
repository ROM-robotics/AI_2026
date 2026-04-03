### detect
```bash
python detect.py --source 0 
```

### recognize
```bash
python .\scripts\recognize.py --mode register --gallery .\data\gallery\
python .\scripts\recognize.py --mode webcam --gallery .\data\gallery\
```

### swap
```bash
python .\scripts\swap.py --source .\data\gallery\hinata.jpg --target 0
```

### detect with liveness
```bash
# Webcam detection + liveness
python detect.py --source 0 --liveness
```

### Recognition with liveness
```bash
python recognize.py --mode webcam --gallery ../data/gallery --liveness
```

### Face swap with liveness
```bash
python swap.py --source ../data/images/source.jpg --target 0 --liveness
```