#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int run(NSString *path, NSString *fnName) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "No Metal device\n");
      return 2;
    }

    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) {
      fprintf(stderr, "Failed to read metallib\n");
      return 2;
    }

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

    id<MTLFunction> fn = [lib newFunctionWithName:fnName];
    if (!fn) {
      fprintf(stderr, "Function not found\n");
      return 2;
    }

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
      fprintf(stderr, "PSO creation failed: %s\n",
              [[error localizedDescription] UTF8String]);
      return 2;
    }

    const NSUInteger N = 256;
    float *host = malloc(N * sizeof(float));
    for (NSUInteger i = 0; i < N; ++i)
      host[i] = 0.0f;

    id<MTLBuffer> buf =
        [device newBufferWithBytes:host
                            length:N * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(N, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd error]) {
      fprintf(stderr, "GPU error: %s\n",
              [[[cmd error] localizedDescription] UTF8String]);
      return 2;
    }

    float *out = (float *)[buf contents];
    int errors = 0;
    for (NSUInteger i = 0; i < N; ++i) {
      if (fabsf(out[i] - 256.0f) > 0.001f) {
        if (errors < 8)
          printf("mismatch[%lu] got=%f expected=256.0\n",
                 (unsigned long)i, out[i]);
        errors++;
      }
    }
    if (errors == 0)
      printf("PASS\n");
    else
      printf("FAIL errors=%d\n", errors);
    free(host);
    return errors == 0 ? 0 : 1;
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: volatile_repro <metallib> <function>\n");
    return 2;
  }
  return run([NSString stringWithUTF8String:argv[1]],
             [NSString stringWithUTF8String:argv[2]]);
}
