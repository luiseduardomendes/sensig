7. Same as Project suggestion 3, but based on video/camera of your laptop/mobile phone and gesture recognition. Commands 'up', 'down', 'left', 'right' could be represented by hand 
   closed to fist with stretched out thump point up, down, left, or right. 'Stop' could be retracting the thump to complete fist. You are free to find unique and easy to recognize 
   gestures.

1.a Use FFT and DR (see Lab), followed by employing at least three different classifier, e.g., kNN, SVM, RBF .... Show 'live classification'

1.b. Compute spectrograms, followed by employing at least three different classifier, e.g., kNN, SVM, RBF .... Show 'live classification'

1.c. Compute spectrograms, develop a compression scheme for spectrograms, employ at least one classifier also used in 1.b. for comparison Show 'live classification'


3. Acoustical mouse guiding project with at least four direction commands recordings, at least 30 per class (command) from different speakers ! (3 persons  on this projectmax.)
   Commands are 'up', 'down', 'left', 'right' to move a pointer, cursor, or a video game entity like PacMan. A 'stop' command could be helpful to avoid to many movement commands, 
   here you flexibility to design the interface.
   Create npy from wav
   Split to train and test sets. 
   Extract the 'Event' and store the reduced data in new npy
   Design a simple python graphics, where the movement of a cursor/pointer according to 'live classification' can be observed and assessed for demo.
   Proceed with feature extraction, dimensionality reduction and classification (hold-out), different approach per participant