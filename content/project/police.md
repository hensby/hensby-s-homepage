+++
# Date this page was created.
date = 2020-01-21T00:00:00

# Project title.
title = "Intelligent case analysis system"

# Project summary to display on homepage.
summary = "Analysis of the police record data get crime name, address, and some other information. Find connection of case."

# Optional image to display on homepage (relative to `static/img/` folder).
image_preview = "static/img/bubbles.jpg"

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["Deep Learning"]

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Optional featured image (relative to `static/img/` folder).
[header]
image = "static/img/headers/bubbles-wide.jpg"
caption = "My caption :smile:"

[design.background]
  # Apply a background color, gradient, or image.
  #   Uncomment (by removing `#`) an option to apply it.
  #   Choose a light or dark text color by setting `text_color_light`.
  #   Any HTML color name or Hex value is valid.
  
  # Background color.
  color = "navy"
  
  # Background gradient.
  gradient_start = "DeepSkyBlue"
  gradient_end = "SkyBlue"
  
  # Background image.
  image = "header/bubbles-wide.jpg"  # Name of image in `static/img/`.
  image_darken = 0.6  # Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.

  # Text color (true=light or false=dark).
  text_color_light = true  
  
+++
-Implemented Named Entity Recognition (NER) Algorithm development. Analysis of the police record data get crime name, address, and some other information. Using LSTM network to Determine entity’s boundaries and CRF to get the Category of entity.

- Implemented Graph Storage and Clustering Algorithms development. To analyze, storage and cluster the entity after NER algorithm analysis. Implement Force Atlas, Spectral clustering and Visualization by Gephi. Using Python, and Neo4j database.


