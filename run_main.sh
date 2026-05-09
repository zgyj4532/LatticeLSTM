python main.py --status train \
		--train ./ResumeNER/train.char.bmes \
		--dev ./ResumeNER/dev.char.bmes \
		--test ./ResumeNER/test.char.bmes \
		--savemodel ./data/resume_ner_model \


# python main.py --status decode \
# 		--raw ./ResumeNER/test.char.bmes \
# 		--savedset ./data/resume_ner_model \
# 		--loadmodel ./data/resume_ner_model.13.model \
# 		--output ./data/resume_ner.raw.out \
