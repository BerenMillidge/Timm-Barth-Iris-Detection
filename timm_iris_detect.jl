
using LinearAlgebra
using Images
using ImageFiltering


function timm_iris_detect(img::AbstractArray{Real,2})
    sigma = 3
    gradx, grady = imgradients(img, KernelFactors.ando3)
    #@assert gradx.size == grady.size, 'Gradient arrays must be of the same size'
    xgradlist = []
    ygradlist = []
    h,w = gradx.size
    for i in 1:h
        for j in 1:w
            x = gradx[i,j]
            y = grady[i,j]
            norm = normalize([x,y])
            push!(xgradlist, norm[1])
            push!(ygradlist, norm[2])
        end
    end
    
    smoothed = imfilter(img, Kernel.Gaussian(sigma))
    center = [h//2, w//2]
    max_sum = -1
    
    for xcenter in 1:h
        for ycenter in 1:w
            current = 0
            for xpos in 1:h
                for ypos in 1:h
                    displacement = normalize([xpos - xcenter, ypos-ycenter])
                    dot = displacement[1] * ygradlist[xpos, ypos] + displacement[2] * xgradlist[xpos, ypos]
                    current += abs(dot)
                end
            end
            weight = smoothed[xcenter, ycenter]
            current_sum *= weight
            
            if current_sum > max_sum
                max_sum = current_sum
                center = [xcenter, ycenter]
            end
        end
    end
    return center          
    
end