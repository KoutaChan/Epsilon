package com.epsilon.prism.cuda;

import com.epsilon.prism.kernel.KernelProvider;
import java.util.ServiceLoader;
import org.testng.Assert;
import org.testng.annotations.Test;

/** GPU を初期化せず、収集用のプロバイダーの登録を確かめる。 */
public class CudaCollectionTest {
  @Test
  public void registersCollectionProvider() {
    var providers =
        ServiceLoader.load(KernelProvider.class).stream()
            .filter(entry -> entry.type() == CudaCollection.class)
            .toList();
    Assert.assertEquals(providers.size(), 1);
    Assert.assertEquals(providers.getFirst().get().backend(), KernelProvider.Backend.CUDA);
  }
}
