.PHONY: %.img %.ecr login

OUTBUCKET = mll-flow-classification

all: clustering.img classification.img

ecr: clustering.ecr classification.ecr

%.ecr: %.img login
	docker tag $(basename $@):latest 463789428463.dkr.ecr.eu-central-1.amazonaws.com/$(basename $@):latest
	docker push 463789428463.dkr.ecr.eu-central-1.amazonaws.com/$(basename $@):latest

%.img:
	docker build -t $(basename $@) ./$(basename $@)

%.run: %.img
	docker run --mount type=bind,source=$$HOME/.aws,target=/root/.aws $(basename $@) $(CMD)

login:
	@eval $$(sudo -u max aws ecr get-login --no-include-email --region eu-central-1)


# s3 functions
sync: # synchronize output folder containg all results
	aws s3 sync s3://$(OUTBUCKET) output
	# aws s3 sync output s3://$(BUCKET)
