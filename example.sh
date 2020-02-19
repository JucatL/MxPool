#PROTEINS -MxPool
python -m train --bmname=PROTEINS --method=MxPool --assign-ratio 0.5 0.1 0.15 --hidden-dim 20 40 60 --output-dim 20 40 60 --cuda=0 --num-classes=2 --num-aspect=3 --multi-conv=1 --multi-pool=1 --max-nodes=200  --epochs=300 --dropout=0.0 --batch-size=20

#COLLAB -MxPool
python -m train --bmname=COLLAB --method=MxPool --assign-ratio 0.15 0.15 0.15 --hidden-dim 30 50 80 --output-dim 30 50 80 --cuda=0 --num-classes=3 --num-aspect=3 --multi-conv=1 --multi-pool=1 --max-nodes=300  --epochs=1000 --dropout=0.0 --batch-size=20

#DD -diffpool
python -m train --bmname=DD --method=diffpool --assign-ratio 0.1 0.1 0.1 --hidden-dim 20 40 60 --output-dim 20 40 60 --cuda=0 --num-classes=2 --num-aspect=3 --multi-conv=1 --multi-pool=1 --max-nodes=500  --epochs=500 --dropout=0.0 --batch-size=20

#ENZYMES -MxPool
python -m train --bmname=ENZYMES --method=MxPool --assign-ratio 0.15 0.15 0.15 --hidden-dim 30 50 80 --output-dim 30 50 80 --cuda=0 --num-classes=6 --num-aspect=3 --multi-conv=1 --multi-pool=1 --max-nodes=100  --epochs=1000 --dropout=0.0 --batch-size=20

#NCI109 -MxPool
python -m train --bmname=NCI109 --method=MxPool --assign-ratio 0.15 0.15 0.15 --hidden-dim 30 50 80 --output-dim 30 50 80 --cuda=1 --num-classes=2 --num-aspect=3 --multi-conv=1 --multi-pool=1 --max-nodes=100  --epochs=1000 --dropout=0.0 --batch-size=20 --lr=0.001
