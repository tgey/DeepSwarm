#!/bin/bash

INSTANCE_NAME=$(grep GCLOUD_INSTANCE_NAME .env | cut -d '=' -f2)
ZONE_NAME=$(grep GCLOUD_INSTANCE_ZONE .env | cut -d '=' -f2)

echo "START SCRIPT"

echo $INSTANCE_NAME
echo $ZONE_NAME

make clean
echo "LOCAL: folders cleaned"

gcloud compute instances start $INSTANCE_NAME --zone=$ZONE_NAME
echo "VM: started"

gcloud compute scp --recurse deepswarm/ $INSTANCE_NAME:~
gcloud compute scp --recurse examples/ $INSTANCE_NAME:~
gcloud compute scp --recurse settings/ $INSTANCE_NAME:~
gcloud compute scp --recurse tests/ $INSTANCE_NAME:~
gcloud compute scp requirements-dev.txt $INSTANCE_NAME:~
gcloud compute scp Makefile $INSTANCE_NAME:~
echo "VM: data transfered from local to instance"

# gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command 'sudo apt install python3-pip'
gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command 'pip3 install -r requirements-dev.txt'
gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command 'ls -la'
echo "VM: command executed"

gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE_NAME
echo "VM: stopped"

notify-send "DeepSwarm: gcloud VM run finished"
echo "END SCRIPT"