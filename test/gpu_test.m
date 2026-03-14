// GPU smoke test: load metallib, create pipeline, dispatch, verify
// Build: clang -framework Metal -framework Foundation gpu_test.m -o gpu_test

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: gpu_test <path.metallib>\n");
        return 1;
    }

    @autoreleasepool {
        // Load device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }
        printf("Device: %s\n", [[device name] UTF8String]);

        // Load metallib
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:argv[1]]];
        if (!data) {
            fprintf(stderr, "Failed to read %s\n", argv[1]);
            return 1;
        }
        printf("Metallib: %lu bytes\n", (unsigned long)[data length]);

        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithData:
            dispatch_data_create([data bytes], [data length],
                nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT)
            error:&error];
        if (!lib) {
            fprintf(stderr, "Failed to load library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }
        printf("Library loaded! Functions:\n");
        for (NSString *name in [lib functionNames])
            printf("  - %s\n", [name UTF8String]);

        // Use the first function found in the library
        NSString *fnName = [[lib functionNames] firstObject];
        if (!fnName) {
            fprintf(stderr, "No functions in library\n");
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:fnName];

        id<MTLComputePipelineState> pso = [device
            newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            fprintf(stderr, "Failed to create PSO: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }
        printf("Pipeline state created! Max threads: %lu\n",
               (unsigned long)[pso maxTotalThreadsPerThreadgroup]);

        // Create buffers
        const int N = 1024;
        float *aData = malloc(N * sizeof(float));
        float *bData = malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) {
            aData[i] = (float)i;
            bData[i] = (float)(N - i);
        }

        id<MTLBuffer> aBuf = [device newBufferWithBytes:aData
            length:N*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:bData
            length:N*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [device newBufferWithLength:N*sizeof(float)
            options:MTLResourceStorageModeShared];

        // Dispatch
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:aBuf offset:0 atIndex:0];
        [enc setBuffer:bBuf offset:0 atIndex:1];
        [enc setBuffer:cBuf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(N / 128, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf error]) {
            fprintf(stderr, "GPU error: %s\n",
                    [[[cmdBuf error] localizedDescription] UTF8String]);
            return 1;
        }

        // Verify
        float *cData = (float *)[cBuf contents];
        int errors = 0;
        for (int i = 0; i < N; i++) {
            float expected = (float)i + (float)(N - i);
            if (fabsf(cData[i] - expected) > 0.001f) {
                if (errors < 5)
                    printf("  MISMATCH [%d]: got %f expected %f\n",
                           i, cData[i], expected);
                errors++;
            }
        }

        if (errors == 0)
            printf("PASS: all %d elements correct!\n", N);
        else
            printf("FAIL: %d/%d errors\n", errors, N);

        free(aData);
        free(bData);
        return errors > 0 ? 1 : 0;
    }
}
