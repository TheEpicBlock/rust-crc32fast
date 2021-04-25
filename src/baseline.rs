use crate::table::CRC32_TABLE;

#[derive(Clone)]
pub struct State {
    state: u32,
    table: [u32; 256],
}

impl State {
    pub fn new(state: u32, polynomial: u32) -> Self {
        State { state, table: create_table(polynomial) }
    }

    pub fn update(&mut self, buf: &[u8]) {
        self.state = update_slow(self.state, buf, &self.table);
    }

    pub fn finalize(self) -> u32 {
        self.state
    }

    pub fn reset(&mut self) {
        self.state = 0;
    }

    pub fn combine(&mut self, other: u32, amount: u64) {
        self.state = crate::combine::combine(self.state, other, amount);
    }
}

pub(crate) fn update_slow(prev: u32, buf: &[u8], table: &[u32; 256]) -> u32 {
    let mut crc = !prev;

    for &byte in buf.iter() {
        crc = (crc << 8) ^ table[((crc >> 24) as u8 ^ byte) as usize];
    }

    !crc
}

pub(crate) fn create_table(polynomial: u32) -> [u32; 256] {
    let mut table = [0u32; 256];

    for i in 0..255 as usize {
        let mut crc = i as u32;

        for j in 0..8 {
            if crc & 0x80 == 0x80 {
                crc = (crc << 1) ^ polynomial;
            } else {
                crc <<= 1;
            }
        }

        table[i] = crc;
    }

    println!("{:#8x}",table[1]);

    return table;
}

#[cfg(test)]
mod test {
    use quickcheck::quickcheck;
    use crate::baseline::create_table;

    #[test]
    fn baseline() {
        let table = create_table(0x04C11DB7);
        assert_eq!(super::update_slow(0, b"", &table), 0);

        // test vectors from the iPXE project (input and output are bitwise negated)
        assert_eq!(super::update_slow(!0x12345678, b"", &table), !0x12345678);
        assert_eq!(super::update_slow(!0xffffffff, b"hello world", &table), !0xf2b5ee7a);
        assert_eq!(super::update_slow(!0xffffffff, b"hello", &table), !0xc9ef5979);
        assert_eq!(super::update_slow(!0xc9ef5979, b" world", &table), !0xf2b5ee7a);

        // Some vectors found on Rosetta code
        //TODO
        // assert_eq!(super::update_slow(0, b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", &table), 0x190A55AD);
        // assert_eq!(super::update_slow(0, b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", &table), 0xFF6CAB0B);
        // assert_eq!(super::update_slow(0, b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F", &table), 0x91267E8A);
    }
}
