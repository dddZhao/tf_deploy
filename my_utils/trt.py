import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from PIL import Image

TRT_LOGGER = trt.Logger()

def preprocess(image):
    # Mean normalization
    image = image.convert("RGB")
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)
    
def postprocess(data):
    num_classes = 21
    # create a color palette, selecting a color for each class
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette*i%255 for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    img = Image.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img
    
def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer_seg(engine, input_file):
    print("Reading input image from file {}".format(input_file))
    with Image.open(input_file) as img:
        orig_size = img.size
        img = img.resize((512,512), Image.Resampling.LANCZOS)
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height

    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference

        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
    img = postprocess(np.reshape(output_buffer, (image_height, image_width))) 
    #with postprocess(np.reshape(output_buffer, (image_height, image_width))) as img:
        #print("Writing output image to file {}".format(output_file))
        #img.convert('RGB').save(output_file, "png")
    return(img,orig_size)

def infer_image(engine, input_file, resize_shape, binding_name="input", input_shape_format="CHW",preprocess_func=None):
    with Image.open(input_file) as img:
        img = img.resize(resize_shape, Image.Resampling.LANCZOS)
        if preprocess_func:
            input_image = preprocess_func(img)
        else:
            input_image = np.asarray(img).astype('float32')
            input_image = np.expand_dims(input_image, axis=0)
        image_width = img.width
        image_height = img.height

    with engine.create_execution_context() as context:
        if input_shape_format == "CHW":
            context.set_binding_shape(engine.get_binding_index(binding_name), (1, 3, image_height, image_width))
        elif input_shape_format == "HWC":
            context.set_binding_shape(engine.get_binding_index(binding_name), (1, image_height, image_width, 3))
        else:
            raise ValueError("Unsupported input_shape_format. Use 'CHW' or 'HWC'.")
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        stream.synchronize()
    return output_buffer

def infer_cla(engine, input_file):
    return infer_image(
        engine, input_file, (299, 299),
        binding_name = "input_1", input_shape_format = "HWC"
    )

def infer_cla2(engine, input_file):
    return infer_image(
        engine, input_file, (224, 224), 
        binding_name = "input", input_shape_format = "CHW", preprocess_func=preprocess
    )

