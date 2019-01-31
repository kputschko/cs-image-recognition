
# Cats v Dogs -------------------------------------------------------------

pacman::p_load(tidyverse, keras)


# Process Data ------------------------------------------------------------
# https://www.kaggle.com/c/dogs-vs-cats/data

# Define raw data path
dir_raw <- "C:/Users/kputs/Downloads/r_data/dogs-vs-cats"

# Define path for subset of all images
dir_small <-
  str_c(dir_raw, "small", sep = "/") %>%
  str_c(c("train", "valid", "test"), sep = "/") %>%
  map(~ str_c(.x, c("cat", "dog"), sep = "/")) %>%
  flatten_chr() %>%
  print()

# Create subset directories
walk(dir_small, ~ dir.create(.x, recursive = TRUE))


# Get filenames and properties of each image
file_train <-
  dir_raw %>%
  str_c("train", sep = "/") %>%
  dir() %>%
  enframe(name = NULL) %>%
  bind_cols(
    dir_raw %>%
      str_c("train", sep = "/") %>%
      dir(full.names = TRUE) %>%
      enframe(name = NULL, value = "dir_input")
  ) %>%
  separate(value, c("animal", "index", "format"), remove = FALSE) %>%
  mutate_at(vars(index), as.integer) %>%
  arrange(animal, index) %>%
  print()


# Split into small train, valid, test sets
file_small_train <-
  file_train %>%
  group_by(animal) %>%
  slice(1:1000) %>%      # 1000 images per animal
  mutate(set = "train",
         dir_output = str_subset(dir_small, pattern = str_glue("{unique(set)}/{unique(animal)}"))) %>%
  print()


file_small_valid <-
  file_train %>%
  group_by(animal) %>%
  slice(1001:1500) %>%   #  500 images per animal
  mutate(set = "valid",
         dir_output = str_subset(dir_small, pattern = str_glue("{unique(set)}/{unique(animal)}"))) %>%
  print()


file_small_test <-
  file_train %>%
  group_by(animal) %>%
  slice(1501:2000) %>%   #  500 images per animal
  mutate(set = "test",
         dir_output = str_subset(dir_small, pattern = str_glue("{unique(set)}/{unique(animal)}"))) %>%
  print()


# Move images into new directories
file_small <-
  bind_rows(file_small_train,
            file_small_valid,
            file_small_test) %>%
  ungroup() %>%
  print()

walk2(
  .x = file_small$dir_input,
  .y = file_small$dir_output,
  .f = ~ file.copy(from = .x, to = .y, overwrite = TRUE))


# Input Data --------------------------------------------------------------

# dir_train_small <-
#   c("C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/train/cat",
#     "C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/train/dog")
#
# dir_valid_small <-
#   c("C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/valid/cat",
#     "C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/valid/dog")
#
# dir_test_small <-
#   c("C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/test/cat",
#     "C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/test/dog")

dir_train_small <- "C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/train"
dir_valid_small <- "C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/valid"
dir_test_small  <- "C:/Users/kputs/Downloads/r_data/dogs-vs-cats/small/test"


# Model Build -------------------------------------------------------------

# Using ImageNet pretrained model
# Include Top = FALSE because we are using our own labels, not ImageNet labels
# Input Shape = ???

conv_base <-
  application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = c(150, 150, 3)
  )

summary(conv_base)

# BUILD MODEL - NEEDS HIGH POWER GPU
model <-
  keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


# We want to use the pre-trained model as is
freeze_weights(conv_base)

# Data augmentation will create more virtual training data based on existing
# training data.  This helps prevent overfitting the model by ensuring the model
# does not look at the same image twice, allowing the model to generalize better

# Don't augment the validation data!

train_datagen <-
  image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE,
    fill_mode = "nearest"
  )

test_datagen <-
  image_data_generator(rescale = 1/255)


train_generator <-
  flow_images_from_directory(
    dir_train_small,            # Target directory
    train_datagen,              # Data generator
    target_size = c(150, 150),  # Resizes all images to 150 × 150
    batch_size = 20,
    class_mode = "binary"       # binary_crossentropy loss for binary labels
  )

validation_generator <-
  flow_images_from_directory(
    dir_valid_small,
    test_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
  )

model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 2e-5),
    metrics = c("accuracy")
  )


#######################
# HEAVY CPU LOAD
#######################
history <-
  model %>%
  fit_generator(
    train_generator,
    steps_per_epoch = 10,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 50
  )


# Test Model --------------------------------------------------------------

test_generator <-
  flow_images_from_directory(
    dir_test_small,
    test_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

#######################
# HEAVY CPU LOAD
#######################
model %>% evaluate_generator(test_generator, steps = 50)

# Save Model --------------------------------------------------------------

model %>% save_model_hdf5("Data/model_cat_dog.h5")


# Load Model --------------------------------------------------------------

model <- load_model_hdf5("Data/model_cat_dog.h5")
