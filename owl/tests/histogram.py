import owl

cpu = owl.create_cpu_device()
owl.set_device(cpu)

x = owl.randn([1000,5000],0,64)
y = x.histogram(8)
print y.to_numpy()
