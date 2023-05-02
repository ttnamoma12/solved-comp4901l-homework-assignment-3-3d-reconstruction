Download Link: https://assignmentchef.com/product/solved-comp4901l-homework-assignment-3-3d-reconstruction
<br>
<h1>Instructions</h1>

Your final upload should have the files arranged in this layout:

&lt;ust login id&gt;.zip

<ul>

 <li>&lt;ust-login-id&gt;/

  <ul>

   <li>&lt;ust-login-id&gt;.pdf</li>

   <li><u>matlab/</u></li>

  </ul></li>

</ul>

∗ camera2.m (provided)

∗ displayEpipolarF.m (provided)

∗ eightpoint.m (Section 3.1.1)

∗ epipolarCorrespondence.m (Section 3.1.2)

∗ epipolarMatchGUI.m (provided)

∗ essentialMatrix.m (Section 3.1.3)

∗ p2t.m (provided)

∗ testDepth.m (provided)

∗ testRectify.m (provided)

∗ refineF.m (provided)

∗ rectify pair.m (Section 3.2.1)

∗ testTempleCoords.m (Section 3.1.5)

∗ triangulate.m (Section 3.1.4)

∗ warp stereo.m (provided)

∗ Any other helper functions you need

<strong>– </strong><u>ec/ </u>(optional for extra credit)

∗ get disparity.m (Section 3.2.2)

∗ get depth.m (Section 3.2.3)

∗ estimate params.m (Section 3.3.2)

∗ estimate pose.m (Section 3.3.1)

∗ projectCAD.m (Section 3.3.3)

∗ testPose.m (provided)

∗ testKRt.m (provided)

<strong>Please make sure you do follow the submission rules mentioned above before uploading your zip file to CASS</strong>. Assignments that violate this submission rule will be <strong>penalized by up to 10% of the total score</strong>.

<ol start="7">

 <li><strong>File paths: </strong>Please make sure that any file paths that you use are relative and not absolute.</li>

</ol>

Not imread(’/name/Documents/subdirectory/hw1/data/xyz.jpg’) but imread(’../data/xyz.jpg’).

<h1>1           Overview</h1>

One of the major areas of computer vision is 3D reconstruction. Given several 2D images of an environment, can we recover the 3D structure of the environment, as well as the position of the camera/robot? This has many uses in robotics and autonomous systems, as understanding the 3D structure of the environment is crucial to navigation. You dont want your robot constantly bumping into walls, or running over human beings!

Figure 1: Example of a robot using SLAM, a 3D reconstruction and localization algorithm

In this assignment there are two programming parts: sparse reconstruction and dense reconstruction. Sparse reconstructions generally contain a number of points, but still manage to describe

Figure 2: The two temple images we have provided to you.

the objects in question. Dense reconstructions are detailed and fine grained. In fields like 3D modelling and graphics, extremely accurate dense reconstructions are invaluable when generating 3D models of real world objects and scenes.

In <strong>Part 1</strong>, you will be writing a set of functions to generate a sparse point cloud for some test images we have provided to you. The test images are 2 renderings of a temple from two different angles. We have also provided you with a mat file containing good point correspondences between the two images. You will first write a function that computes the fundamental matrix between the two images. Then write a function that uses the epipolar constraint to find more point matches between the two images. Finally, you will write a function that will triangulate the 3D points for each pair of 2D point correspondences.

We have provided you with a few helpful mat files. someCorresps.mat contains good point correspondences. You will use this to compute the Fundamental matrix. intrinsics.mat contains the intrinsic camera matrices, which you will need to compute the full camera projection matrices. Finally templeCoords.mat contains some points on the first image that should be easy to localize in the second image.

In <strong>Part 2</strong>, we utilize the extrinsic parameters computed by Part 1 to further achieve dense 3D reconstruction of this temple. You will need to compute the rectification parameters. We have provided you with testRectify.m (and some helper functions) that will use your rectification function to warp the stereo pair. You will then <em>optionally </em>use the warped pair to compute a disparity map and finally a dense depth map.

In both cases, multiple images are required, because without two images with a large portion overlapping the problem is mathematically underspecified. It is for this same reason biologists suppose that humans, and other predatory animals such as eagles and dogs, have two front facing eyes. Hunters need to be able to discern depth when chasing their prey. On the other hand herbivores, such as deer and squirrels, have their eyes position on the sides of their head, sacrificing most of their depth perception for a larger field of view. The whole problem of 3D reconstruction is inspired by the fact that humans and many other animals rely on dome degree of depth perception when navigating and interacting with their environment. Giving autonomous systems this information is very useful.

Figure 3: Camera and light source triangle diagram

<h1>2           Theory</h1>

<h2>2.1         Triangulation (5 points)</h2>

Structured light is a general way of retrieving 3D structure from a stationary camera. One very interesting way to do this is by using a light source and a thin shadow (maybe cast by a pencil).

See http://www.vision.caltech.edu/bouguetj/ICCV98/ for an example.

This scheme uses a stationary lamp, and a stationary camera. Then a thin wand is waved above the object, and a shadow plane is cast. From the camera view, you know the angle to a point on the object is <em>β</em>, and from the position of the wand, you know the angle at which the shadow is cast from the light source is <em>α </em>You also know how far the camera is from the light source, so you know the baseline distance <em>b</em>. Given this information, how can you recover the position of the point in the scene? In other words, can you represent the coordinates (<em>x,z</em>) of the point <em>P </em>by only <em>b,α,β</em>?

<h2>2.2         Fundamental Matrix (15 points)</h2>

Suppose two cameras fixate on a point <em>P </em>(see Figure 4) in space such that their optical axes intersect at that point. Show that if the image coordinates are normalized so that the coordinate origin (0<em>,</em>0) coincides with the principal point, the <strong>F</strong><sub>33 </sub>element of the fundamental matrix <strong>F </strong>is zero.

Figure 4: Figure for Question 2.2. <em>C</em><sub>1 </sub>and <em>C</em><sub>2 </sub>are the optical centers. The principal axes intersect at point <em>P</em>.

<h1>3           Programming</h1>

<h2>3.1         Sparse Reconstruction</h2>

In this section, you will be writing a set of function to compute the sparse reconstruction from two sample images of a temple. You will first estimate the Fundamental matrix, compute point correspondences, then plot the results in 3D. It may be helpful to read through Section 3.1.5 right now. In Section 3.1.5 we ask you to write a testing script that will run your whole pipeline. It will be easier to start that now and add to it as you complete each of the questions one after the other.

<h3>3.1.1         Implement the eight point algorithm:           (10 points)</h3>

In this question, youre going to use the eight point algorithm which is covered in class to estimate the fundamental matrix. Please use the point correspondences provided in someCorresp.mat. Write a function with the following signature: function F = eightpoint(pts1, pts2, M)

where x1 and x2 are <em>N </em>×2 matrices corresponding to the (<em>x,y</em>) coordinates of the <em>N </em>points in the first and second image respectively. <em>M </em>is a scale parameter.

<ul>

 <li>Normalize points and un-normalize F: You should scale the data by dividing each coordinate by M (the maximum of the images width and height). After computing F, you will have to “unscale” the fundamental matrix.</li>

 <li>You must enforce the rank 2 constraint on <strong>F </strong>before unscaling. Recall that a valid fundamental matrix <strong>F </strong>will have all epipolar lines intersect at a certain point, meaning that there exists a non-trivial null space for <strong>F</strong>. In general, with real points, the eightpoint solution for <strong>F </strong>will not come with this condition. To enforce the rank 2 condition, decompose <strong>F </strong>with SVD to get the three matrices <strong>U</strong>, <strong>Σ</strong>, <strong>V </strong>such that <strong>F </strong>= <strong>UΣV</strong><em><sup>T</sup></em>. Then force the matrix to be rank 2 by setting the smallest singular value in <strong>Σ </strong>to zero, giving you a new <strong>Σ</strong><sup>0</sup>. Now compute the proper fundamental matrix with <strong>F</strong><sup>0 </sup>= <strong>UΣ</strong><sup>0</sup><strong>V</strong><em><sup>T</sup></em>.</li>

 <li>You may find it helpful to refine the solution by using local minimization. This probably wont fix a completely broken solution, but may make a good solution better by locally minimizing a geometric cost function. For this we have provided m (takes in Fundamental matrix and the two sets of points), which you can call from eightpoint before unscaling <strong>F</strong>. This function uses matlab’s fminsearch to non-linearly search for a better <strong>F </strong>that minimizes the cost function. For this to work, it needs an initial guess for <strong>F </strong>that is already close to the minimum.</li>

 <li>Remember that the <em>x</em>-coordinate of a point in the image is its column entry and ycoordinate is the row entry. Also note that eight-point is just a figurative name, it just means that you need at least 8 points; your algorithm should use an over-determined system (<em>N &gt; </em>8 points).</li>

 <li>To test your estimated <strong>F</strong>, use the provided function m (takes in <strong>F </strong>and the two images). This GUI lets you select a point in one of the images and visualize the corresponding epipolar line in the other image like in Figure 5</li>

</ul>

<strong>In your write-up</strong>: Please include your recovered <strong>F </strong>and the visualization of some epipolar lines (similar to Figure 5).

Figure 5: Epipolar lines visualization from displayEpipolarF.m

<h3>3.1.2         Find epipolar correspondences:       (20 points)</h3>

To reconstruct a 3D scene with a pair of stereo images, we need to find many point pairs. A point pair is two points in each image that correspond to the same 3D scene point. With enough of these pairs, when we plot the resulting 3D points, we will have a rough outline of the 3D object. You found point pairs in the previous homework using feature detectors and feature descriptors, and testing a point in one image with every single point in the other image. But here we can use the fundamental matrix to greatly simplify this search.

Figure 6: Epipolar Geometry (source Wikipedia)

Recall from class that given a point <em>x </em>in one image (the left view in Figure 6). Its corresponding 3D scene point <em>p </em>could lie anywhere along the line from the camera center <em>o </em>to the point <em>x</em>. This line, along with a second image’s camera center <em>o</em><sup>0 </sup>(the right view in Figure 6) forms a plane. This plane intersects with the image plane of the second camera, resulting in a line <em>l</em><sup>0 </sup>in the second image which describes all the possible locations that <em>x </em>may be found in the second image. Line <em>l</em><sup>0 </sup>is the epipolar line, and we only need to search along this line to find a match for point <em>x </em>found in the first image.

Write a function with the following signature:

function pts2 = epipolarCorrespondence(im1, im2, F, pts1)

where im1 and im2 are the two images in the stereo pair. F is the fundamental matrix computed for the two images using your eightpoint function. pts1 is an <em>N </em>×2 matrix containing the (<em>x,y</em>) points in the first image. Your function should return pts2, an <em>N </em>× 2 matrix, which contains the corresponding points in the second image.

<ul>

 <li>To match one point <em>x </em>in image 1, use fundamental matrix to estimate the corresponding epipolar line <em>l</em><sup>0 </sup>and generate a set of candidate points in the second image.</li>

 <li>For each candidate points <em>x</em><sup>0</sup>, a similarity score between <em>x </em>and <em>x</em><sup>0 </sup>is computed. The point among candidates with highest score is treated as epipolar correspondence.</li>

 <li>There are many ways to define the similarity between two points. Feel free to use whatever you want and <strong>describe it in your write-up</strong>. One possible solution is to select a small window of size <em>w </em>around the point <em>x</em>. Then compare this target window to the window of the candidate point in the second image. For the images we gave you, simple Euclidean distance or Manhattan distance should suffice. Manhattan distance was not covered in class. Consider Googling it.</li>

 <li>Remember to take care of data type and index range.</li>

</ul>

You can use epipolarMatchGui.m to visually test your function. Your function does not need to be perfect, but it should get most easy points correct, like corners, dots etc…

<strong>In your write-up</strong>: Include a screen shot of epipolarMatchGui running with your implementation of epipolarCorrespondence. Mention the similarity metric you decided to use. Also comment on any cases where your matching algorithm consistently fails, and why you might think this is.

<h3>3.1.3         Write a function to compute the essential matrix: (10 points)</h3>

In order to get the full camera projection matrices we need to compute the Essential matrix.

So far, we have only been using the Fundamental matrix. Write a function with the following signature: function E = essentialMatrix(F, K1, K2)

where F is the Fundamental matrix computed between two images, K1 and K2 are the intrinsic camera matrices for the first and second image respectively (contained in intrinsics.mat). E is the computed essential matrix. The intrinsic camera parameters are typically acquired through camera calibration.

Refer to the class slides for the relationship between the Fundamental matrix and the Essential matrix.

<strong>In your write-up: </strong>Write your estimated E matrix for the temple image pair we gave you.

<h3>3.1.4         Implement triangulation:     (20 points)</h3>

Write a function to triangulate pairs of 2D points in the images to a set of 3D points with the signature:

function pts3d = triangulate(P1, pts1, P2, pts2) where pts1 and pts2 are the <em>N </em>× 2 matrices with the 2D image coordinates and <em>pts</em>3<em>d </em>is an <em>N </em>× 3 matrix with the corresponding 3D points (in all cases, one point per row). P1 and P2 are the 3×4 camera projection matrices. Remember that you will need to multiply the given intrinsic matrices with your solution for the extrinsic camera matrices to obtain the final camera projection

Figure 7: Epipolar Match visualization. A few errors are alright, but it should get most easy points correct (corners, dots, etc…)

matrices. For P1 you can assume no rotation or translation, so the extrinsic matrix is just [<strong>I</strong>|<strong>0</strong>]. For P2, pass the essential matrix to the provided camera2.m function to get four possible extrinsic matrices. You will need to determine which of these is the correct one to use (see hint in Section

3.1.5).

Refer to the class slides for one possible triangulation algorithm. Once you have it implemented, check the performance by looking at the re-projection error. To compute the re-projection error, project the estimated 3D points back to the image 1(2) and compute the mean Euclidean error between projected 2D points and pts1(2).

<strong>In your write-up: </strong>Describe how you detemined which extrinsic matrices is correct. Note that simply rewording the hint is not enough. Report your re-projection error using pts1, pts2 from someCorresp.mat. If implemented correctly, the re-projection error should be less than 1 pixel.

<h3>3.1.5         Write a test script that uses templeCoords:               (10 points)</h3>

You now have all the pieces you need to generate a full 3D reconstruction. Write a test script testTempleCoords.m that does the following:

<ol>

 <li>Load the two images and the point correspondences from mat</li>

 <li>Run eightpoint to compute the fundamental matrix F</li>

 <li>Load the points in image 1 contained in mat and run your epipolarCorrespondences on them to get the corresponding points in image 2</li>

 <li>Load mat and compute the essential matrix E.</li>

 <li>Compute the first camera projection matrix <strong>P</strong><sub>1 </sub>and use m to compute the four candidates for <strong>P</strong><sub>2</sub></li>

</ol>

Figure 8: Sample Reconstructions

<ol start="6">

 <li>Run your triangulate function using the four sets of camera matrix candidates, the points from mat and their computed correspondences. 7. Figure out the correct <strong>P</strong><sub>2 </sub>and the corresponding 3D points.</li>

</ol>

Hint: You’ll get 4 projection matrix candidates for camera 2 from the essential matrix. The correct configuration is the one for which most of the 3D points are in front of both cameras (positive depth).

<ol start="7">

 <li>Use matlab’s plot3 function to plot these point correspondences on screen</li>

 <li>Save your computed rotation matrix (R1, R2) and translation (t1, t2) to the file</li>

</ol>

../data/extrinsics.mat. These extrinsic parameters will be used in the next section.

We will use your test script to run your code, so be sure it runs smoothly. In particular, use relative paths to load files, not absolute paths.

<strong>In your write-up</strong>: Include 3 images of your final reconstruction of the templeCoords points, from different angles.

<h2>3.2         Dense Reconstruction</h2>

In applications such as 3D modelling, 3D printing, and AR/VR, a sparse model is not enough. When users are viewing the reconstruction, it is much more pleasing to deal with a dense reconstruction. To do this, it is helpful to rectify the images to make matching easier. In this section, you will be writing a set of functions to perform a dense reconstruction on our temple examples. Given the provided intrinsic and computed extrinsic parameters, you will need to write a function to compute the rectification parameters of the two images. The rectified images are such that the epipolar lines are horizontal, so searching for correspondences becomes a simple linear. This will be done for every point. Finally, you can optionally compute the disparity and depth map.

<h3>3.2.1         Image Rectification   (10 points)</h3>

Write a program that computes rectification matrices.

function [M1,M2,K1n,K2n,R1n,R2n,t1n,t2n] = rectify pair (K1,K2,R1,R2,t1,t2)

This function takes left and right camera parameters (K,R,t) and returns left and right rectification matrices (M1,M2) and updated camera parameters. You can test your function using the provided script testRectify.m.

From what we learned in class, the rectify pair function should consecutively run the following steps:

<ol>

 <li>Compute the optical center <strong>c</strong><sub>1 </sub>and <strong>c</strong><sub>2 </sub>of each camera by <strong>c </strong>= −(<strong>KR</strong>)<sup>−1</sup>(<strong>Kt</strong>).</li>

 <li>Compute the new rotation matrix <strong>R</strong><sub>e </sub>= [<strong>r</strong><sub>1 </sub><strong>r</strong><sub>2 </sub><strong>r</strong><sub>3</sub>]<em><sup>T </sup></em>where <strong>r</strong><sub>1</sub><em>,</em><strong>r</strong><sub>2</sub><em>,</em><strong>r</strong><sub>3 </sub>∈ R<sup>3×1 </sup>are orthonormal vectors that represent <em>x</em>-, <em>y</em>-, and <em>z</em>-axes of the camera reference frame, respectively.

  <ul>

   <li>The new <em>x</em>-axis (<strong>r</strong><sub>1</sub>) is parallel to the baseline: <strong>r</strong><sub>1 </sub>= (<strong>c</strong><sub>1 </sub>− <strong>c</strong><sub>2</sub>)<em>/</em>||<strong>c</strong><sub>1 </sub>− <strong>c</strong><sub>2</sub>||.</li>

   <li>The new <em>y</em>-axis (<strong>r</strong><sub>2</sub>) is orthogonal to <em>x </em>and to any arbitrary unit vector, which we set to be the <em>z </em>unit vector of the old left matrix: <strong>r</strong><sub>2 </sub>is the cross product of <strong>R</strong><sub>1</sub>(3<em>,</em>&#x1f642;<em><sup>T </sup></em>and <strong>r</strong><sub>1</sub>.</li>

   <li>The new <em>z</em>-axis (<strong>r</strong><sub>3</sub>) is orthogonal to <em>x </em>and <em>y</em>: <strong>r</strong><sub>3 </sub>is the cross product of <strong>r</strong><sub>2 </sub>and <strong>r</strong><sub>1</sub>.</li>

  </ul></li>

 <li>Compute the new intrinsic parameter <strong>K</strong><sub>e </sub>. We can use an arbitrary one. In our test code, we just let .</li>

 <li>Compute the new translation: <strong>t</strong><sub>1 </sub>= −<strong>Rc</strong><sub>e 1</sub>, <strong>t</strong><sub>2 </sub>= −<strong>Rc</strong><sub>e 2</sub>.</li>

 <li>Finally, the rectification matrix of the first camera can be obtained by</li>

</ol>

<strong>M</strong>1 = (<strong>K</strong>e1<strong>R</strong>e1)(<strong>KR</strong>)−1                                                                                                        (1)

<strong>M</strong><sub>2 </sub>can be computed from the same formula.

Once you have finished, run testRectify.m (Be sure to have the extrinsics saved by your testTempleCoords.m). This script will test your rectification code on the temple images using the provided intrinsic parameters and your computed extrinsic paramters. It will also save the estimated rectification matrix and updated camera parameters in temple.mat, which will be used by the next test script testDepth.m.

<strong>In your write-up</strong>: Include a screen shot of the result of running testRectify.m on temple images. The results should show some epipolar lines that are perfectly horizontal, with corresponding points in both images lying on the same line.

<h3>3.2.2         Dense window matching to find per pixel disparity (extra credit)              (20 points)</h3>

Write a program that creates a disparity map from a pair of rectified images (im1 and im2).

function dispM = get disparity(im1,im2,maxDisp,windowSize) where maxDisp is the maximum disparity and windowSize is the window size. The output dispM has the same dimension as im1 and im2. Since im1 and im2 are rectified, computing correspondences is reduced to a 1-D search problem.

The dispM(y, x) is

dispM(<em>y,x</em>) = arg          min            dist(im1(<em>y,x</em>)<em>,</em>im2(<em>y,x </em>− <em>d</em>))<em>,                                       </em>(2)

0≤<em>d</em>≤maxDisp

where dist(im1(<em>y,x</em>)<em>,</em>im2im2(<em>y </em>+ <em>i,x </em>+ <em>j </em>− <em>d</em>))<sup>2 </sup>with <em>w </em>is (windowSize − 1)<em>/</em>2. This summation on the window can be easily computed by using the conv2 Matlab function (i.e. convolve with a mask of ones(windowSize,windowSize)) Note that this is not the only way to implement this.

<h3>3.2.3         Depth map (extra credit)       (10 points)</h3>

Write a function that creates a depth map from a disparity map (dispM).

function depthM = get depth(dispM,K1,K2,R1,R2,t1,t2)

Use the fact that depthM(<em>y,x</em>) = <em>b </em>× <em>f/</em>dispM(<em>y,x</em>) where <em>b </em>is the baseline and <em>f </em>is the focal length of camera. For simplicity, assume that <em>b </em>= ||<em>c</em><sub>1 </sub>− <em>c</em><sub>2</sub>|| (i.e., distance between optical centers) and <em>f </em>= <em>K</em><sub>1</sub>(1<em>,</em>1). Finally, let depthM(<em>y,x</em>) = 0 whenever dispM(<em>y,x</em>) = 0 to avoid dividing by 0.

You can now test your disparity and depth map functions using testDepth.m. Be sure to have the rectification saved (by running testRectify.m). This function will rectify the images, then compute the disparity map and the depth map.

<strong>In your write-up</strong>: Include images of your disparity map and your depth map.

<h2>3.3         Pose Estimation (Extra Credit)</h2>

In this section, you will implement what you have learned on class to estimate both the intrinsic and extrinsic parameters of camera given 2D points <strong>x </strong>on image and their corresponding 3D points <strong>X</strong>. In other words, given a set of matched points {<strong>X</strong><em><sub>i</sub>,</em><strong>x</strong><em><sub>i</sub></em>} and camera model

(<strong>X</strong>;<strong>p</strong>) = <em> ,                                                                     </em>(3)

we want to find the estimate of the camera matrix <strong>P </strong>∈ R<sup>3×4</sup>, as well as intrinsic parameter matrix <strong>K </strong>∈ R<sup>3×3</sup>, camera rotation <strong>R </strong>∈ R<sup>3×3 </sup>and camera translation <strong>t </strong>∈ R<sup>3</sup>, such that

<strong>P </strong>= <strong>K</strong>[<strong>R</strong>|<strong>t</strong>]<em>.                                                                                      </em>(4)

<h3>3.3.1         Estimate camera matrix P     (10 points)</h3>

Write a function that estimates the camera matrix <strong>P </strong>given 2D and 3D points <strong>x</strong><em>,</em><strong>X</strong>.

function P = estimate pose(x, X),

where <strong>x </strong>is 2 × <em>N </em>matrix denoting the (<em>x,y</em>) coordinates of the <em>N </em>points on the image plane and <strong>X </strong>is 3 × <em>N </em>matrix denoting the (<em>x,y,z</em>) coordinates of the corresponding points in the 3D world. Recall that this camera matrix can be computed using the same strategy as homography estimation by Direct Linear Transform (DLT). Once you finish this function, you can run the provided script testPose.m to test your implementation.

<strong>In your write-up</strong>: Include the output of the script testPose.

<h3>3.3.2         Estimate intrinsic/extrinsic parameters     (20 points)</h3>

Write a function that estimates both intrinsic and extrinsic parameters from camera matrix.

function [K, R, t] = estimate params(P)

From what we learned on class, the estimate params should consecutively run the following steps:

<ol>

 <li>Compute the camera center <strong>c </strong>by using SVD. Hint: <strong>c </strong>is the eigenvector corresponding to the smallest eigenvalue.</li>

 <li>Compute the intrinsic <strong>K </strong>and rotation <strong>R </strong>by using QR decomposition. <strong>K </strong>is a right upper triangle matrix while <strong>R </strong>is a orthonormal matrix. Checking this answer<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> might help.</li>

 <li>Compute the translation by <strong>t </strong>= <strong>Rc</strong>.</li>

</ol>

Once you finish your implementation, you can run the provided script testKRt.m. <strong>In your write-up</strong>: Include the output of the script testKRt.

<h3>3.3.3         Project a CAD model to the image    (20 points)</h3>

Now you will utilize what you have implemented to estimate the camera matrix from a real image,shown in Figure 9(left), and project the 3D object (CAD model), shown in Figure 9(right), back on to the image plane.

Figure 9: The provided image and 3D CAD model

Write a script projectCAD.m that does the following:

<ol>

 <li>Load an image image, a CAD model cad, 2D points x and 3D points X from mat.</li>

 <li>Run estimate pose and estimate params to estimate camera matrix P, intrinsic matrix K, rotation matrix R, and translation t.</li>

 <li>Use your estimated camera matrix P to project the given 3D points X onto the image.</li>

 <li>Plot the given 2D points x and the projected 3D points on screen. An example is shown in Figure 10(left). Hint: use plot.</li>

 <li>Draw the CAD model rotated by your estimated rotation R on screen. An example is shown in Figure 10(middle). Hint: use trimesh.</li>

 <li>Project the CAD’s all vertices onto the image and draw the projected CAD model overlapping with the 2D image. An example is shown in Figure 10(right). Hint: use patch.</li>

</ol>

<strong>In your write-up</strong>: Include the three images similar to Figure 10. You have to use different colors from Figure 10. For example, green circle for given 2D points, black points for projected 3D points, blue CAD model, and red projected CAD model overlapping on the image. You will get <strong>NO </strong>credit if you use the same color.

Figure 10: Project a CAD model back onto the image. Left: the image annotated with given 2D points (blue circle) and projected 3D points (red points); Middle: the CAD model rotated by estimated R; Right: the image overlapping with projected CAD model.

<a href="#_ftnref1" name="_ftn1">[1]</a> http://math.stackexchange.com/questions/1640695/rq-decomposition