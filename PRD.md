# Objective
Turn sheet music in PDF form into Google Slides
# Key Results
- Each PDF page has a top and bottom section of the score. Create an image of each section with the dimensions of one Google slide
- You should create images for each **included** half-page section (after dropping non-music pages and empty halves) and put them in order
- Put those images into Google Slides in order
# Final Deliverable
A set of music score images in Google Slides
# Success Criteria
- The Google Slides contain all **included** half-page sections in order (blank pages and empty halves are dropped; **text-only title pages are not detected in v1**)
- The section image should take up the entire slide
- Slide order matches **document order** among included sections only (dropped pages do not leave gaps in the sense of placeholder slides—they are omitted entirely)
# Suggested Plan
Break the task into two subtasks:
- First, create the images from the PDF file
- Then, put the images into Google Slides

# Decisions (clarifications)
- **Page split:** Fixed **50%** horizontal split (top half / bottom half of each page).
- **Slide fit:** Images are **scaled to fill** the default slide (16:9). Cropping is allowed **only** to remove whitespace/margins; **no cropping through music notation** (staff lines, notes, symbols must remain fully visible). Implementation likely needs a **content bounds** step on each half-page image so “fill” does not cut into the score.
- **What to include:** **Skip** blank pages and **empty** half-sections after the 50% split. **No special “title page” rule** (do not skip page 1 by position). *Later:* optionally detect **text-only** pages and drop them.
- **Google Slides:** **Create a new presentation** on each run. Authenticate with a **personal Google account** via **OAuth** (not a service account).
- **Product shape:** **Local script / CLI** (no web app required for v1).
- **Input PDFs:** **Scanned** sheet music is in scope (expect raster pages; handle resolution/skew pragmatically as needed).