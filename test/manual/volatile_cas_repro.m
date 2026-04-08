#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: volatile_cas_repro <metallib>\n");
    return 2;
  }

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "No Metal device\n");
      return 2;
    }

    NSData *data = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:argv[1]]];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:
        dispatch_data_create([data bytes], [data length], nil,
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT)
                                              error:&error];
    if (!lib) {
      fprintf(stderr, "Library load failed: %s\n",
              [[error localizedDescription] UTF8String]);
      return 2;
    }

    id<MTLFunction> fn =
        [lib newFunctionWithName:@"volatile_cas_repro"];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
      fprintf(stderr, "PSO creation failed: %s\n",
              [[error localizedDescription] UTF8String]);
      return 2;
    }

    int32_t flagInit = 0;
    float dataInit = 0.0f;
    float outInit = -1.0f;

    id<MTLBuffer> flagBuf =
        [device newBufferWithBytes:&flagInit length:sizeof(flagInit)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dataBuf =
        [device newBufferWithBytes:&dataInit length:sizeof(dataInit)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf =
        [device newBufferWithBytes:&outInit length:sizeof(outInit)
                           options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:flagBuf offset:0 atIndex:0];
    [enc setBuffer:dataBuf offset:0 atIndex:1];
    [enc setBuffer:outBuf offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(2, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd error]) {
      fprintf(stderr, "GPU error: %s\n",
              [[[cmd error] localizedDescription] UTF8String]);
      return 2;
    }

    float out = *(float *)[outBuf contents];
    int32_t flag = *(int32_t *)[flagBuf contents];
    float finalData = *(float *)[dataBuf contents];
    if (fabsf(out - 42.0f) > 0.001f) {
      printf("FAIL out=%f flag=%d data=%f\n", out, flag, finalData);
      return 1;
    }

    printf("PASS\n");
    return 0;
  }
}
