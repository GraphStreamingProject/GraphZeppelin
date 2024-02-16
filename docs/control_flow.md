# GraphZeppelin Control Flow
These charts describe how our basic operations are performed in GraphZeppelin.

## Driver Level Flow
---

### Initialization
```mermaid
flowchart TD
    A[User] -->|2. Construct| B[GraphSketchDriver]
    A -->|1. Construct| C
    B -->|3. get_num_vertices\n4. get_desired_update_batch\n5. allocate_worker_memory\n6. print_configuration| C[Sketch Algorithm]
```

### Stream Processing
```mermaid
flowchart TD
    A[User] -->|1. process_stream_until| B[GraphSketchDriver]
    B -->|5. resume| E[WorkerThreadGroup]
    B -->|4. insert| D[GutteringSystem]
    E -->|5. get_data| D
    E -->|6. apply_update_batch| C
    B -->|3. pre_insert| C[Sketch Algorithm]
```

### Preforming a Query
```mermaid
flowchart TD
    A[User] -->|1. prep_query| B[GraphSketchDriver]
    A -->|5. query| C
    B -->|2. has_cached_query| C[Sketch Algorithm]
    B -->|3. flush| D[GutteringSystem]
    B -->|4. pause| E[WorkerThreadGroup]
```
