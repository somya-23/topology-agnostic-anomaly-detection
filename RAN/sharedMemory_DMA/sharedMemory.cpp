#include <iostream>
#include <fcntl.h>      // O_* flags
#include <sys/mman.h>   // shm_open, mmap
#include <unistd.h>     // close

#define SHM_NAME "/my_shared_memory"
#define SHM_SIZE 1024

// Define the same struct for key-value pairs
struct KeyValuePair {
    char key[50];
    char value[50];
};

// Define shared memory structure
struct SharedDictionary {
    int count;  // Number of key-value pairs
    KeyValuePair entries[5];
};

int main() {
    // Open existing shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 2412);
    if (shm_fd == -1) {
        perror("shm_open failed");
        return 1;
    }

    // Map shared memory
    void* ptr = mmap(0, SHM_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed");
        return 1;
    }

    // Cast to SharedDictionary struct
    SharedDictionary* dict = static_cast<SharedDictionary*>(ptr);

    // Print key-value pairs
    std::cout << "Reader: Received Dictionary from Shared Memory\n";
    for (int i = 0; i < dict->count; i++) {
        std::cout << dict->entries[i].key << " => " << dict->entries[i].value << std::endl;
    }

    // Cleanup
    munmap(ptr, SHM_SIZE);
    close(shm_fd);

    return 0;
}
