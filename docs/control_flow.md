# GraphZeppelin Control Flow
These charts describe how our basic operations are performed in GraphZeppelin.

## Driver Level Flow

### Initialization
The driver pulls information from the graph sketch algorithm and tells the algorithm to allocate scratch space for its threads to use.
```mermaid
flowchart TD
    A[User] -->|2. Construct| B[GraphSketchDriver]
    A -->|1. Construct| C
    B -->|3. get_num_vertices\n4. get_desired_update_batch\n5. allocate_worker_memory\n8. print_configuration| C[Sketch Algorithm]
    B -->|6. Construct| D[GutteringSystem]
    B -->|7. Construct| E[WorkerThreadGroup]
```

### Stream Processing
When processing a stream, the driver coordinates its own threads, the `GutteringSystem` which batches updates, the `WorkerThreadGroup` which applies sketch updates, and the graph sketch algorithm. Once the setup steps 1-2 complete, for each stream update until the breakpoint (either query or end of stream) we perform steps 3-7.
```mermaid
flowchart TD
    A[User] -->|1. process_stream_until| B[GraphSketchDriver]
    B -->|2. resume| E[WorkerThreadGroup]
    B -->|4. insert| D[GutteringSystem]
    E -->|5. get_data| D
    E -->|6. batch_callback| B
    B --->|3. pre_insert\n7. apply_update_batch| C[Sketch Algorithm]
```

### Preforming a Query
To perform a query, the user must first call `driver.prep_query()` in which the driver ensures the query is safe to perform. Specifically, the driver must ensure that all stream updates have been processed before allowing the query to continue. If step 2 `has_cached_query()` returns true, the driver can safely skip steps 3-4 and immediately allow the user to perform the query.
```mermaid
flowchart TD
    A[User] -->|1. prep_query| B[GraphSketchDriver]
    A -->|5. query| C
    B -->|2. has_cached_query| C[Sketch Algorithm]
    B -->|3. flush| D[GutteringSystem]
    B -->|4. pause| E[WorkerThreadGroup]
```
