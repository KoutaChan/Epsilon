package com.epsilon.reviewer.store;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.zip.Deflater;
import java.util.zip.GZIPOutputStream;

/** 新しいファイルに gzip（Deflater のレベル 9）で書く。閉じるときに gzip の末尾を書き、ディスクに同期する。 */
final class SyncedGzipFile extends GZIPOutputStream {
  private static final int BUFFER = 64 * 1024;
  private final FileChannel channel;
  private boolean closed;

  private SyncedGzipFile(FileChannel channel) throws IOException {
    super(new BufferedOutputStream(Channels.newOutputStream(channel), BUFFER), BUFFER);
    this.channel = channel;
    def.setLevel(Deflater.BEST_COMPRESSION);
  }

  static SyncedGzipFile create(Path path) throws IOException {
    FileChannel channel =
        FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
    try {
      return new SyncedGzipFile(channel);
    } catch (IOException | RuntimeException failure) {
      channel.close();
      throw failure;
    }
  }

  @Override
  public void close() throws IOException {
    if (closed) return;
    closed = true;
    try {
      finish();
      out.flush();
      channel.force(true);
    } finally {
      def.end();
      channel.close();
    }
  }
}
