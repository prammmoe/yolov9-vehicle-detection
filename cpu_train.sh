python train_dual.py --workers 4 --device cpu --batch 16 --data vehicle_datasets/vehicle.yaml --img 640 --cfg models/detect/yolov9-e.yaml --weights '' --name yolov9-e.pt --hyp hyp.scratch-high.yaml --min-items 0 --epochs 5 --close-mosaic 15