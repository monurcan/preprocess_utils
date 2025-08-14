Currently there are 2 things need to do:
1. Image alignment: Lez tried already (series_aligned folders), but the alignment is not very good (e.g., building 324, building304).
    Solution correspondence matching models like lightglue
    subpixel matching models?
    use the first frame to find correspondence, not previous one?
    ransac like outlier handling methods, moving objects like leaves effects the transformation predicted
    lez method probably finds 
    then dont use correspondences corresponding to the moving objects 
    since we aligned now we can eliminate them
    
    LEZ SAID WE SHOULD ALIGN ALL THE OBJECTS INCLUDING THE MOVING ONES! SO MAYBE NON HOMOGRAPHIC, MORE COMPLEX TRANSFORMATIONS LIKE https://graphics.cs.wisc.edu/Papers/2009/LGJA09/siggraph09_preprint.pdf CAN BE HELPFUL!

    ASK: I think image size should be the same for all images. I understand why you cropped for extreme transformations (like building304), but reflection also looks natural.

2. Create mask for sky: because the cloud is varied across different images. 