// https://github.com/jeffhammond/dpcpp-tutorial
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <exception>

using namespace sycl;

constexpr int N = 42;

int main(){
    // I use array on the stack because the array is not
    // huge, only 42 elements. I would advise to use std::vector
    // when the numbers of elements is massive or exceed the stack
    // size of 1 MB
	std::array<int, N> a, b;
    // Initialize a and b array.
	for(int i=0; i< N; ++i) {
		a[i] = i;
		b[i] = 0;
	}
	
	try {
		queue Q(host_selector{}); // Select to run on the CPU
		// Initialize the buffer variables for the device with
        // host array.
		buffer<int, 1> A{a.data(), range<1>(a.size())};
		buffer<int, 1> B{b.data(), range<1>(b.size())};
		
		Q.submit([&](handler& h) {
            // Device will access the arrays through accA and accB
			auto accA = A.template get_access<access::mode::read>(h);
			auto accB = B.template get_access<access::mode::write>(h);
			h.parallel_for<class nstream>(
				range<1>{N},
				[=](id<1> i) { accB[i] = accA[i]; });
			
		});
        // Wait for the device code to complete
		Q.wait();

        // Synchronize the device's B array to host's b array by reading it
        // If Q.wait() is not called previously, B.get_access() will call
        // it behind the scene before the synchronize.
		B.get_access<access::mode::read>(); // <--- Host Accessor to Synchronize Memory
		for(int i=0; i< N; ++i) {
			std::cout << b[i] << " ";
		}
	}
	catch(sycl::exception& ex)
	{
		std::cerr << "SYCL Exception thrown: " << ex.what() << std::endl;
	}
	catch(std::exception& ex)
	{
		std::cerr << "std Exception thrown: " << ex.what() << std::endl;
	}
	std::cout << "\nDone!\n";
	return 0;
}
