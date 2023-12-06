# study_course_project
This is study project developed during the studying at Karpov.courses

The project's essence is to implement a service that will, at any given moment, return posts for each user to be displayed in their social media feed. Essentially, it involves building a recommendation system for posts within the social network.

The project includes:
- Working with a database
- Creating features/feature engineering and a training dataset
- Training the model and  its evaluation on a validation set
- The actual service itself: loading the model -> obtaining features for the model based on user_id -> predicting posts that the user will like -> returning the result.
- A/B testing features established as well (model_control and model_test -> are different models to tests)
