# Third-party licenses

License texts for the third-party code and libraries used by this project.

| Component | License | License files |
|---|---|---|
| [Nyanten (Cryolite)](https://github.com/Cryolite/nyanten) | MIT; some upstream tests use GPL-3.0-or-later | [LICENSE](nyanten/LICENSE.md) |
| [Shanten Number (tomohxx)](https://github.com/tomohxx/shanten-number) | LGPL-3.0 | [LGPL](shanten-number/COPYING.LESSER), [GPL](../LICENSE) |
| DJL / djl-rocm | Apache-2.0 | [LICENSE](djl/LICENSE), [NOTICE](djl/NOTICE) |
| Gson | Apache-2.0 | [LICENSE](gson/LICENSE) |
| Error Prone annotations | Apache-2.0 | [COPYING](error-prone/COPYING) |
| JNA | Apache-2.0 | [LICENSE](jna/LICENSE), [Apache-2.0](jna/AL2.0) |
| libffi (included in JNA) | MIT | [LICENSE](jna/libffi-LICENSE) |
| Apache Commons Compress | Apache-2.0 | [LICENSE](commons-compress/LICENSE.txt), [NOTICE](commons-compress/NOTICE.txt) |
| Apache Commons Codec | Apache-2.0 | [LICENSE](commons-codec/LICENSE.txt), [NOTICE](commons-codec/NOTICE.txt) |
| Apache Commons IO | Apache-2.0 | [LICENSE](commons-io/LICENSE.txt), [NOTICE](commons-io/NOTICE.txt) |
| zstd-jni | BSD-2-Clause | [LICENSE](zstd-jni/LICENSE) |
| Zstandard / xxHash (included in zstd-jni) | BSD-3-Clause | [LICENSE](zstandard/LICENSE), [xxHash NOTICE](zstandard/xxHash-NOTICE.txt) |
| SLF4J | MIT | [LICENSE](slf4j/LICENSE.txt) |
| Logback classic / core | LGPL-2.1 | [LICENSE](logback/LICENSE.txt), [LGPL](logback/LGPL-2.1.txt), [EPL](logback/EPL-1.0.html) |
| PyTorch | BSD-3-Clause | [LICENSE](pytorch/LICENSE) |
| {fmt} / hipBLASLt / origami / yaml-cpp | MIT | [fmt](rocm7.2/fmt-LICENSE), [hipBLASLt](rocm7.2/hipblaslt-LICENSE.md), [origami](rocm7.2/origami-LICENSE.md), [yaml-cpp](rocm7.2/yaml-cpp-LICENSE) |
| msgpack | Boost-1.0 | [COPYING](rocm7.2/msgpack-COPYING), [LICENSE](rocm7.2/msgpack-LICENSE_1_0.txt) |
| Maven Wrapper (build tool) | Apache-2.0 | [LICENSE](maven-wrapper/LICENSE), [NOTICE](maven-wrapper/NOTICE) |

Where a license choice is available, this project uses Apache-2.0 for JNA, LGPL-2.1 for Logback, and BSD-3-Clause for Zstandard.

The PyTorch and ROCm component licenses apply when those components are included in the distribution.
