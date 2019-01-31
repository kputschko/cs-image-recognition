
# Image Recognition -------------------------------------------------------

# Courtesy of
# https://blogs.rstudio.com/tensorflow/posts/2018-09-07-getting-started/

pacman::p_load(tidyverse, keras)
pacman::p_load_gh("rstudio/tensorflow")

# Load TensorFlow + Keras -------------------------------------------------
# Python 3.6 needs to be installed

install_tensorflow()
install_keras()

# Load Data ---------------------------------------------------------------

data_fashion <- keras::dataset_fashion_mnist()

c(train_images, train_labels) %<-% data_fashion$train
c(test_images, test_labels) %<-% data_fashion$test


# Labels ------------------------------------------------------------------

class_names <-  c('T-shirt/top',
                  'Trouser',
                  'Pullover',
                  'Dress',
                  'Coat',
                  'Sandal',
                  'Shirt',
                  'Sneaker',
                  'Bag',
                  'Ankle boot')


# Data Processing ---------------------------------------------------------

image_1 <- as_tibble(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank(),
        aspect.ratio = 1) +
  labs(x = NULL, y = NULL)


train_images <- train_images / 255
test_images <- test_images / 255


par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}


# Model -------------------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(train_images, train_labels, epochs = 5)

score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")


# Predict -----------------------------------------------------------------

predictions <- model %>% predict(test_images)

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

test_labels[1]

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev))
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}

img <- test_images[1, , , drop = FALSE]
dim(img)

predictions_1 <- model %>% predict(img)
predictions_1

prediction_1 <- predictions_1[1, ] - 1
which.max(prediction_1)

class_pred <- model %>% predict_classes(img)
class_pred
