Stranger Than Fiction
=====================

> “Truth is stranger than fiction, but it is because Fiction is obliged
> to stick to possibilities; Truth isn't.”
>
> Mark Twain

Project for autonomous navigation ground truth generation and
analysis.

Libraries and Subdirectories
----------------------------
- apriltag: The APRIL ground truthing tag library from APRIL and Ed
  Olson. Original can be found at
  http://april.eecs.umich.edu/wiki/index.php/AprilTags

- gt: Higher level library for returning poses from images that may or
  may not contain images. Designed to use different AR tag detectors.

- analysis: Python scripts to generate metrics and plots for comparing
  estimated poses to ground truth poses.
