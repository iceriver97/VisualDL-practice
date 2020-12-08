from ppgan.apps import DeOldifyPredictor
deoldify = DeOldifyPredictor()
pred = deoldify.run_image("../imgs/test_old.jpg")
pred.save('deoldify_result.jpg')