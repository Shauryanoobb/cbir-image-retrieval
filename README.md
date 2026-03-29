This is the current implementation, one feature, one distance metric, one evaluation metric,
later we'll make this code base generilsable (where we can use call the entire pipeline with any feature, any distance metric and any evaluation metric )
repo structure as of now
cbir/
│
├── dataset/                  # download Wang dataset (1000 images) (this should have 1000 jpg files)
│
├── feature_extraction.py     # color histogram for now
├── similarity.py             # distance function, euclidian for now
├── retrieval.py              # retrieval logic (right now we're building the database each time)
├── evaluation.py             # precision@K for now
├── visualise.py              # to see what results we got on the query image
├── main.py                   # run everything, make it configurable later
│
└── requirements.txt

how wang dataset is organised rn-> 10 categories, each category has 100 images, so look at the first digit on ABC.png for the category
0xx.jpg → Class 0 (Africans)
1xx.jpg → Class 1 (Beaches)
2xx.jpg → Class 2 (Buildings)
...
9xx.jpg → Class 9 (Food)